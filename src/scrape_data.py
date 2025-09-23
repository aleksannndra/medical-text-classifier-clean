# src/scrape_data.py

import os
import time
import requests
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract

BASE_URL = "https://www.mp.pl"
START_URL = "https://www.mp.pl/pacjent/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_DIR = "data/raw"

def get_category_links(start_url):
    """Get dictionary of {category_name: url} for all categories on mp.pl/pacjent"""
    response = requests.get(start_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    header5 = soup.find("h5", string="Serwisy specjalistyczne")
    ul = header5.find_next_sibling("ul")

    category_links = {}
    for a in ul.find_all("a", href=True):
        name = a.get_text(strip=True)
        url = BASE_URL + a["href"]
        category_links[name] = url
    return category_links

def get_article_links(category_url):
    """Extract all article links from a given category page"""
    response = requests.get(category_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    article_urls = set()

    # top article
    top_article = soup.find("a", class_="top-link", href=True)
    if top_article:
        article_urls.add(BASE_URL + top_article["href"])

    # other articles
    for article in soup.find_all("article", class_="other-article"):
        link = article.find("a", href=True)
        if link:
            article_urls.add(BASE_URL + link["href"])

    return article_urls

def save_article(text, category_name, idx):
    """Save article text into raw data folder"""
    folder = os.path.join(OUTPUT_DIR, category_name)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"{category_name}_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def process_category(category_name, category_url):
    """Scrape all articles from one category"""
    print(f"Processing category: {category_name}")
    article_urls = get_article_links(category_url)

    for idx, url in enumerate(article_urls):
        html = fetch_url(url)
        if not html:
            continue
        article_text = extract(html, favor_recall=True, include_tables=False)
        if article_text:
            save_article(article_text, category_name, idx)
        else:
            time.sleep(1)

def main():
    categories = get_category_links(START_URL)
    for name, url in categories.items():
        process_category(name, url)

if __name__ == "__main__":
    main()

