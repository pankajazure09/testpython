import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
from pypdf import PdfReader


logger = logging.getLogger(__name__)

CHUNK_CHAR_LENGTH = int(os.getenv("PDF_CHUNK_CHAR_LENGTH", "1800"))
CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "200"))
SUMMARY_CHAR_LIMIT = int(os.getenv("PDF_SUMMARY_CHAR_LIMIT", "16000"))

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

_blob_service_client: Optional[BlobServiceClient] = None
_openai_client: Optional[AzureOpenAI] = None
_search_client: Optional[SearchClient] = None


@dataclass
class RequestPayload:
    user_id: str
    file_name: str
    pdf_base64: str

    @classmethod
    def from_request(cls, req: func.HttpRequest) -> "RequestPayload":
        try:
            body = req.get_json()
        except ValueError as exc:
            raise ValueError("Request body must be valid JSON.") from exc

        missing = [field for field in ["userId", "fileName", "pdfBase64"] if field not in body]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

        if not isinstance(body["pdfBase64"], str):
            raise ValueError("Field 'pdfBase64' must be a base64-encoded string.")

        return cls(
            user_id=str(body["userId"]),
            file_name=str(body["fileName"]),
            pdf_base64=body["pdfBase64"],
        )


def get_blob_service_client() -> BlobServiceClient:
    global _blob_service_client
    if _blob_service_client is None:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is not configured.")
        _blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return _blob_service_client


def upload_pdf_to_blob_storage(pdf_bytes: bytes, file_name: str) -> str:
    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "pdf-documents")
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    sanitized_file_name = (file_name or "document.pdf").replace(" ", "-")
    name_root, ext = os.path.splitext(sanitized_file_name)
    if not ext:
        ext = ".pdf"
    unique_blob_name = f"{name_root}-{uuid.uuid4().hex}{ext}"
    blob_client = container_client.get_blob_client(unique_blob_name)

    blob_client.upload_blob(pdf_bytes, overwrite=False)
    return blob_client.url


def decode_pdf(payload: RequestPayload) -> bytes:
    try:
        return base64.b64decode(payload.pdf_base64, validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 content provided for 'pdfBase64'.") from exc


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    text_parts: List[str] = []
    for page_number, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to read page %s: %s", page_number, exc)
            page_text = ""
        if page_text:
            text_parts.append(page_text.strip())
    text = "\n\n".join(text_parts).strip()
    if not text:
        raise ValueError("Unable to extract text from the provided PDF.")
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_CHAR_LENGTH, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than the chunk size.")

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = max(end - overlap, 0)
        if start == 0 and end == text_length:
            break

    return chunks


def get_openai_client() -> AzureOpenAI:
    global _openai_client
    if _openai_client is None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        if not endpoint or not api_key:
            raise EnvironmentError("Azure OpenAI endpoint and key must be configured.")
        _openai_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    return _openai_client


def generate_text_embeddings(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not model:
        raise EnvironmentError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured.")

    response = get_openai_client().embeddings.create(model=model, input=chunks)
    embeddings: List[Optional[List[float]]] = [None] * len(chunks)
    for item in response.data:
        embeddings[item.index] = item.embedding

    if any(embedding is None for embedding in embeddings):
        raise RuntimeError("Failed to retrieve embeddings for all chunks.")

    return embeddings  # type: ignore


def generate_insights(text: str) -> str:
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not model:
        raise EnvironmentError("AZURE_OPENAI_CHAT_DEPLOYMENT is not configured.")

    truncated_text = text if len(text) <= SUMMARY_CHAR_LIMIT else text[:SUMMARY_CHAR_LIMIT]
    response = get_openai_client().chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that reads PDF documents and produces high-quality insights. "
                    "Provide a concise summary, list key points, and suggest potential follow-up actions."
                ),
            },
            {"role": "user", "content": truncated_text},
        ],
    )
    try:
        return response.choices[0].message.content.strip()
    except (IndexError, AttributeError):
        raise RuntimeError("Failed to generate insights from Azure OpenAI.")


def get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        index_name = os.getenv("AZURE_SEARCH_INDEX")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        if not endpoint or not index_name or not api_key:
            raise EnvironmentError("Azure Cognitive Search configuration is incomplete.")
        _search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
    return _search_client


def _document_id(user_id: str, file_name: str, index: int) -> str:
    safe_user_id = user_id.replace(" ", "-")
    safe_file_name = file_name.replace(" ", "-")
    return f"{safe_user_id}-{safe_file_name}-{index}"


def upsert_vector_documents(
    chunks: List[str],
    embeddings: List[List[float]],
    user_id: str,
    file_name: str,
    blob_url: str,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings counts do not match.")

    documents = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        documents.append(
            {
                "id": _document_id(user_id, file_name, idx),
                "userId": user_id,
                "fileName": file_name,
                "blobPath": blob_url,
                "chunkIndex": idx,
                "content": chunk,
                "contentVector": embedding,
            }
        )

    results = get_search_client().upload_documents(documents)
    failures = [result for result in results if not result.succeeded]
    if failures:
        errors = ", ".join(f"{failure.key}: {failure.error_message}" for failure in failures)
        raise RuntimeError(f"Failed to upload one or more documents to the vector store: {errors}")


def response_json(payload: dict, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(payload),
        status_code=status_code,
        mimetype="application/json",
    )


@app.function_name(name="process_pdf")
@app.route(route="process-pdf", methods=[func.HttpMethod.POST])
def process_pdf(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = RequestPayload.from_request(req)
        pdf_bytes = decode_pdf(payload)
        blob_url = upload_pdf_to_blob_storage(pdf_bytes, payload.file_name)
        text = extract_text_from_pdf(pdf_bytes)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No textual content available to generate embeddings from the PDF.")
        embeddings = generate_text_embeddings(chunks)
        upsert_vector_documents(chunks, embeddings, payload.user_id, payload.file_name, blob_url)
        insights = generate_insights(text)

        return response_json(
            {
                "message": "PDF processed successfully.",
                "insights": insights,
                "chunkCount": len(chunks),
                "blobUrl": blob_url,
            },
            status_code=200,
        )
    except ValueError as exc:
        logger.exception("Validation error while processing PDF.")
        return response_json({"error": str(exc)}, status_code=400)
    except EnvironmentError as exc:
        logger.exception("Configuration error while processing PDF.")
        return response_json({"error": str(exc)}, status_code=500)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error while processing PDF.")
        return response_json({"error": str(exc)}, status_code=500)