# crawler.py


#--------------------------------------------WORKING CRAWLER------------------------------------------------------------
#----------------------------------------ALREADY SAVED CRAWLED DATA IN 'data/' FOLDER-----------------------------------
import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin, urlparse

os.makedirs("data", exist_ok=True)

START_URLS = [
    "https://www.changiairport.com/",
    "https://www.jewelchangiairport.com/"
]

visited = set()

def is_internal_link(base_url, link):
    parsed_base = urlparse(base_url)
    parsed_link = urlparse(link)
    return parsed_link.netloc == "" or parsed_link.netloc == parsed_base.netloc

def scrape_recursive(base_url, url, depth=1, max_depth=3):
    if depth > max_depth or url in visited:
        return ""
    visited.add(url)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ""
    except:
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    clean_text = re.sub(r'\s+', ' ', text)

    # Find all internal links
    for link in soup.find_all('a', href=True):
        href = urljoin(base_url, link['href'])
        if is_internal_link(base_url, href):
            clean_text += " " + scrape_recursive(base_url, href, depth + 1, max_depth)

    return clean_text

for start_url in START_URLS:
    visited.clear()
    print(f"Scraping {start_url} ...")
    all_text = scrape_recursive(start_url, start_url, depth=1, max_depth=3)
    domain_name = urlparse(start_url).netloc.replace(".", "_")
    with open(f"data/{domain_name}.txt", "w", encoding="utf-8") as f:
        f.write(all_text)

print("Full site scraping complete. Data saved in 'data/' folder.")
