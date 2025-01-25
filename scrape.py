import requests
from bs4 import BeautifulSoup
import json
import time
import random
from fake_useragent import UserAgent
from urllib.parse import quote_plus
import backoff
import logging
from datetime import datetime
import os

# Create logs directory if it doesn't exist
log_dir = os.path.join('log', 'scrape')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AmazonProductScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.database = []
        self.seen_products = set()
        self.total_products_scraped = 0

    def get_headers(self):
        """Generate random headers for each request"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'TE': 'Trailers'
        }

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, Exception),
        max_tries=3
    )
    def fetch_page(self, url, page=1):
        """Fetch page with retry logic"""
        try:
            full_url = f"{url}&page={page}" if page > 1 else url
            logger.info(f"Fetching page {page}: {full_url}")
            
            response = self.session.get(
                full_url,
                headers=self.get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            content_length = len(response.text)
            logger.info(f"Successfully fetched page {page} - Content length: {content_length}")
            
            return response.text
        except Exception as e:
            logger.error(f"Error fetching page {page} - {url}: {str(e)}")
            raise

    def extract_product_details(self, item):
        """Extract product details from a search result item"""
        try:
            asin = item.get('data-asin')
            if not asin:
                asin = item.get('data-component-id')
            if not asin or asin in self.seen_products:
                return None

            title_elem = (
                item.select_one('span.a-text-normal') or 
                item.select_one('h2 a span') or
                item.select_one('.a-link-normal .a-text-normal')
            )
            
            price_elem = (
                item.select_one('span.a-price span.a-offscreen') or
                item.select_one('span.a-price:not(.a-text-price) span.a-offscreen') or
                item.select_one('.a-price .a-offscreen')
            )
            
            rating_elem = (
                item.select_one('span.a-icon-alt') or
                item.select_one('.a-star-rating-text') or
                item.select_one('.a-icon-star-small')
            )
            
            reviews_elem = (
                item.select_one('span.a-size-base.s-underline-text') or
                item.select_one('.a-size-base.s-underline-text') or
                item.select_one('.a-link-normal .a-size-base')
            )
            
            image_elem = (
                item.select_one('img.s-image') or
                item.select_one('.s-image') or
                item.select_one('.a-link-normal img')
            )

            if not title_elem:
                return None

            title = title_elem.text.strip()
            product_url = f"https://www.amazon.com/dp/{asin}"

            if not any(keyword.lower() in title.lower() for keyword in ['skin', 'face', 'cream', 'serum', 'moisturizer', 'treatment']):
                return None

            price = None
            if price_elem:
                try:
                    price_text = price_elem.text.strip().replace('$', '').replace(',', '')
                    price = float(price_text)
                except (ValueError, AttributeError):
                    logger.debug(f"Could not parse price for {asin}")

            rating = None
            if rating_elem:
                try:
                    rating_text = rating_elem.text.split(' ')[0]
                    rating = float(rating_text)
                except (ValueError, IndexError, AttributeError):
                    logger.debug(f"Could not parse rating for {asin}")

            reviews_count = None
            if reviews_elem:
                try:
                    reviews_text = reviews_elem.text.replace(',', '').replace('(', '').replace(')', '')
                    reviews_count = int(''.join(filter(str.isdigit, reviews_text)))
                except (ValueError, AttributeError):
                    logger.debug(f"Could not parse reviews count for {asin}")

            image_url = image_elem.get('src') if image_elem else None

            self.seen_products.add(asin)
            
            logger.debug(f"Successfully extracted product: {title[:50]}...")
            return {
                "title": title,
                "product_url": product_url,
                "price": price,
                "rating": rating,
                "reviews_count": reviews_count,
                "image_url": image_url
            }

        except Exception as e:
            logger.error(f"Error extracting product details: {str(e)}")
            return None

    def scrape_products(self, category, keywords, num_products=50):
        products = []
        search_query = quote_plus(" ".join(keywords + ["skincare"]))
        base_url = f"https://www.amazon.com/s?k={search_query}&ref=nb_sb_noss"
        
        logger.info(f"Starting scrape for category '{category}' with keywords: {keywords}")
        
        page = 1
        while len(products) < num_products and page <= 7:  # Limit to 7 pages
            logger.info(f"Scraping page {page} for category '{category}' - Products found so far: {len(products)}/{num_products}")
            
            html_content = self.fetch_page(base_url, page)
            if not html_content:
                logger.warning(f"No content retrieved for page {page}")
                break
                
            soup = BeautifulSoup(html_content, 'html.parser')
            items = soup.select('div[data-asin]:not([data-asin=""]), .s-result-item[data-asin]:not([data-asin=""]), .sg-col-inner')
            
            logger.info(f"Found {len(items)} potential items on page {page}")
            
            for item in items:
                if len(products) >= num_products:
                    break
                    
                product = self.extract_product_details(item)
                if product:
                    products.append(product)
                    self.total_products_scraped += 1
                    logger.info(f"Added product {len(products)}/{num_products}: {product['title'][:50]}...")
                
                time.sleep(random.uniform(0.5, 1.0))
            
            page += 1
            time.sleep(random.uniform(2, 3))

        logger.info(f"Completed scraping for category '{category}' - Total products found: {len(products)}")
        return products

    def create_database(self):
        categories = [
            {
                "category": "Skincare for Fair Skin",
                "keywords": ["fair skin", "pale skin", "light skin", "sensitive"]
            },
            {
                "category": "Skincare for Medium Skin",
                "keywords": ["medium skin", "normal skin", "combination skin"]
            },
            {
                "category": "Skincare for Dark Skin",
                "keywords": ["dark skin", "melanin", "hyperpigmentation"]
            },
            {
                "category": "Acne Treatment",
                "keywords": ["acne", "blemish", "breakout", "pimple"]
            },
            {
                "category": "Aging Treatment",
                "keywords": ["anti aging", "wrinkle", "fine lines", "mature skin"]
            },
            {
                "category": "Dry Skin Treatment",
                "keywords": ["dry skin", "dehydrated", "moisturizing"]
            },
            {
                "category": "Oily Skin Treatment",
                "keywords": ["oily skin", "excess oil", "shine control"]
            },
            {
                "category": "Sensitive Skin Treatment",
                "keywords": ["sensitive skin", "gentle", "soothing", "calming"]
            },
            {
                "category": "Combination Skin Treatment",
                "keywords": ["combination skin", "t-zone", "balanced"]
            },
            {
                "category": "Texture Treatment",
                "keywords": ["uneven texture", "rough skin", "smoothing"]
            }
        ]

        start_time = time.time()
        logger.info("Starting database creation")
        
        for idx, category_info in enumerate(categories, 1):
            try:
                logger.info(f"\nProcessing category {idx}/{len(categories)}: {category_info['category']}")
                products = self.scrape_products(
                    category_info['category'],
                    category_info['keywords']
                )
                
                if products:
                    self.database.append({
                        "category": category_info['category'],
                        "keywords": category_info['keywords'],
                        "products": products
                    })
                    
                    logger.info(f"Added {len(products)} products to {category_info['category']}")
                    self.save_database()  # Save after each category
                    
                time.sleep(random.uniform(3, 5))
                
            except Exception as e:
                logger.error(f"Error processing category {category_info['category']}: {str(e)}")
                continue

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"\nDatabase creation completed in {duration:.2f} seconds")
        logger.info(f"Total products scraped: {self.total_products_scraped}")

    def save_database(self):
        try:
            with open('product_database.json', 'w', encoding='utf-8') as f:
                json.dump(self.database, f, indent=2, ensure_ascii=False)
            logger.info("Database saved successfully")
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")

    def load_existing_database(self, filename):
        """Load existing database from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            logger.info(f"Successfully loaded existing database from {filename}")
        except FileNotFoundError:
            logger.warning(f"No existing database found at {filename}")
            self.database = []
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            self.database = []

    def deduplicate_products(self, products):
        """Remove duplicates keeping ones with more reviews"""
        title_map = {}
        for product in products:
            title = product['title'].lower().strip()
            if title in title_map:
                existing_reviews = title_map[title].get('reviews_count', 0) or 0
                current_reviews = product.get('reviews_count', 0) or 0
                if current_reviews > existing_reviews:
                    title_map[title] = product
            else:
                title_map[title] = product
        return list(title_map.values())

    def fill_to_fifty(self, existing_products, category, keywords):
        """Fill product list up to 50 items"""
        needed_products = 50 - len(existing_products)
        if needed_products <= 0:
            return existing_products

        logger.info(f"Needs {needed_products} more products for {category}")
        new_products = self.scrape_products(category, keywords, needed_products)
        combined_products = existing_products + new_products
        return self.deduplicate_products(combined_products)

    def plus_fifty(self, existing_products, category, keywords):
        """Add 50 more products to existing list"""
        new_products = self.scrape_products(category, keywords, 50)
        combined_products = existing_products + new_products
        return self.deduplicate_products(combined_products)

def main():
    try:
        print("\nAmazon Product Scraper")
        print("1) Rescrape (New 50 products per category)")
        print("2) Fill to 50 (Complete categories with less than 50 products)")
        print("3) Plus 50 (Add 50 more products to each category)")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")

        scraper = AmazonProductScraper()

        if choice in ['2', '3']:
            while True:
                input_file = input("\nEnter the name of the existing database file: ").strip()
                if input_file.endswith('.json'):
                    try:
                        scraper.load_existing_database(input_file)
                        break
                    except FileNotFoundError:
                        print(f"File {input_file} not found. Please try again.")
                else:
                    print("Please enter a .json file name.")

        output_file = input("\nEnter name for the output database file: ").strip()
        if not output_file.endswith('.json'):
            output_file += '.json'

        if choice == '1':
            # Complete rescrape
            scraper.create_database()
        else:
            # Process existing database
            for category_data in scraper.database:
                category = category_data['category']
                keywords = category_data['keywords']
                existing_products = category_data['products']

                logger.info(f"\nProcessing {category}")
                logger.info(f"Current products: {len(existing_products)}")

                if choice == '2':
                    # Fill to 50
                    category_data['products'] = scraper.fill_to_fifty(
                        existing_products, category, keywords
                    )
                else:  # choice == '3'
                    # Plus 50
                    category_data['products'] = scraper.plus_fifty(
                        existing_products, category, keywords
                    )

                logger.info(f"Final products for {category}: {len(category_data['products'])}")
                
                # Save after each category
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(scraper.database, f, indent=2, ensure_ascii=False)
                    logger.info(f"Progress saved to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving progress: {str(e)}")

        logger.info("\nProcess completed successfully")
        logger.info(f"Final database saved as: {output_file}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
