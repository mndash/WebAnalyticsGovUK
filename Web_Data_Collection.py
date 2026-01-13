
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time
import logging
import os

# Configure logging
logging.basicConfig(
    filename='scraping_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

main_site = 'https://www.gov.uk'
start_url = "https://www.gov.uk/search/guidance-and-regulation?organisations[]=marine-management-organisation&amp;order=most-viewed"
store_path = "/mnt/lab/unrestricted/muhammed.njie@marinemanagement.org.uk/published/"
parquet_path = f"{store_path}mmo_published_guidance.parquet"

def get_next_page(soup):
    nav = soup.find("div", class_="govuk-pagination__next")
    if nav:
        page = nav.find("a", class_="govuk-link govuk-pagination__link")
        if page:
            return urljoin(main_site, page.get('href'))
    return None

def fetch_publication(url, existing_links):
    all_data = []
    page_count = 0

    while url:
        try:
            response = requests.get(url, verify=False, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            articles = soup.find_all("li", class_="gem-c-document-list__item")

            for article in articles:
                title_div = article.find('div', class_="gem-c-document-list__item-title")
                date_tag = article.find('time')
                if title_div:
                    link = title_div.find("a")
                    if link:
                        title = link.text.strip()
                        full_url = urljoin(main_site, link.get('href'))
                        if full_url in existing_links:
                            continue
                        date = date_tag.text.strip() if date_tag else "N/A"
                        all_data.append({"Title": title, "URL": full_url, "Last Updated": date})

            page_count += 1
            print(f"Page {page_count} scraped successfully.")
            time.sleep(0.125)  # Rate limit: 8 requests per second

            url = get_next_page(soup)

        except requests.exceptions.RequestException as e:
            error_message = f"Error fetching page {page_count + 1}: {e}"
            print(error_message)
            logging.error(error_message)
            break

    print("Published Guidance Scraping completed.")
    return all_data

if __name__ == "__main__":
    # Ensure directory exists
    dbutils.fs.mkdirs(f"dbfs:{store_path}")

    # Load existing data if parquet exists
    try:
        existing_df = pd.read_parquet(f"/dbfs{parquet_path}")
        existing_links = set(existing_df["URL"].tolist())
    except Exception:
        existing_df = pd.DataFrame(columns=["Title", "URL", "Last Updated"])
        existing_links = set()

    # Fetch new data
    new_data = fetch_publication(start_url, existing_links)
    new_df = pd.DataFrame(new_data)

    # Combine and remove duplicates
    combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["URL"])

    # Save back to parquet
    combined_df.to_parquet(f"/dbfs{parquet_path}", engine="pyarrow", index=False)
    print(f"Total records after update: {len(combined_df)}")
