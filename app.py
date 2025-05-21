import requests
response = requests.get("https://www.sci.gov.in/?_siwp_captcha&id=bs8j0nf8fnfswdq2031t716uas59pkaclu5jaqey")
print(response.status_code)
print(response.content)