import requests
from bs4 import BeautifulSoup

# Target URL
url = "https://woocommercegst.co.in/"  # Replace with any URL

# Headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Send request
response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}")

# If request successful
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Print all text content from paragraphs and headings
    print("\n📄 Page Text:\n")
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
        text = tag.get_text(strip=True)
        if text:
            print(f"- {text}")

    # Print all image URLs
    print("\n🖼️ Image URLs:\n")
    for img in soup.find_all("img"):
        src = img.get("src")
        if src:
            print(f"- {src}")

else:
    print("❌ Failed to fetch page")
