from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import random
import time
from fake_useragent import UserAgent
from urllib.parse import quote_plus
import backoff
import logging
from ml_model.face_analyzer import FaceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Analysis API",
    description="API for analyzing facial features and recommending skincare products",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

face_analyzer = FaceAnalyzer()

class ProductDetails(BaseModel):
    title: str
    product_url: str
    price: Optional[float] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    image_url: Optional[str] = None

    def dict(self, *args, **kwargs):
        return {
            "title": self.title,
            "product_url": self.product_url,
            "price": self.price,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
            "image_url": self.image_url
        }

class ProductRecommendation(BaseModel):
    category: str
    keywords: List[str]
    products: List[ProductDetails]

class FullResponse(BaseModel):
    analysis: dict
    recommendations: List[ProductRecommendation]

class AmazonScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.recommended_products = set()  # Track recommended products to avoid duplicates

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
    def fetch_page(self, url):
        """Fetch page with retry logic"""
        try:
            response = self.session.get(
                url,
                headers=self.get_headers(),
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Successfully fetched page: {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {url}: {str(e)}")
            raise

    def extract_product_details(self, item):
        """Extract product details from a search result item"""
        try:
            logger.debug(f"Processing item: {item}")

            asin = item.get('data-asin')
            if not asin:
                asin = item.get('data-component-id')
            if not asin:
                return None

            product_url = f"https://www.amazon.com/dp/{asin}"
            
            if product_url in self.recommended_products:
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
                logger.debug(f"No title found for ASIN: {asin}")
                return None

            title = title_elem.text.strip()
            
            price = None
            if price_elem:
                try:
                    price_text = price_elem.text.strip().replace('$', '').replace(',', '')
                    price = float(price_text)
                except (ValueError, AttributeError):
                    logger.debug(f"Could not parse price for {asin}")
                    pass

            rating = None
            if rating_elem:
                try:
                    rating_text = rating_elem.text.split(' ')[0]
                    rating = float(rating_text)
                except (ValueError, IndexError, AttributeError):
                    logger.debug(f"Could not parse rating for {asin}")
                    pass

            reviews_count = None
            if reviews_elem:
                try:
                    reviews_text = reviews_elem.text.replace(',', '').replace('(', '').replace(')', '')
                    reviews_count = int(''.join(filter(str.isdigit, reviews_text)))
                except (ValueError, AttributeError):
                    logger.debug(f"Could not parse reviews count for {asin}")
                    pass

            image_url = image_elem.get('src') if image_elem else None

            product = ProductDetails(
                title=title,
                product_url=product_url,
                price=price,
                rating=rating,
                reviews_count=reviews_count,
                image_url=image_url
            )

            logger.debug(f"Successfully extracted product: {product}")
            return product

        except Exception as e:
            logger.error(f"Error extracting product details: {str(e)}")
            return None

    def search_products(self, search_terms: List[str], num_products: int = 5) -> List[ProductDetails]:
        try:
            search_query = quote_plus(" ".join(search_terms))
            url = f"https://www.amazon.com/s?k={search_query}&ref=nb_sb_noss"
            
            logger.info(f"Searching Amazon with URL: {url}")
            
            html_content = self.fetch_page(url)
            
            logger.debug(f"Retrieved HTML content length: {len(html_content)}")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            products = []
            processed_count = 0
            
            product_containers = (
                soup.select('div[data-asin]:not([data-asin=""])') or
                soup.select('.s-result-item[data-asin]:not([data-asin=""])') or
                soup.select('.sg-col-inner')
            )
            
            logger.info(f"Found {len(product_containers)} potential product containers")
            
            for item in product_containers:
                if processed_count >= num_products:
                    break
                    
                product = self.extract_product_details(item)
                if product:
                    products.append(product)
                    self.recommended_products.add(product.product_url)
                    processed_count += 1
                    logger.debug(f"Added product: {product.title}")
                    
                time.sleep(random.uniform(0.5, 1.0))
            
            logger.info(f"Successfully extracted {len(products)} products")
            return products
            
        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            return []

amazon_scraper = AmazonScraper()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing services...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Face Analysis API is running"}

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    response_timeout = 120
    try:
        max_size = 5 * 1024 * 1024  # 5MB
        contents = await file.read()
        if len(contents) > max_size:
            return JSONResponse(
                status_code=413,
                content={"detail": "File too large. Maximum size is 5MB."}
            )

        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=415,
                content={"detail": "File type not supported. Please upload an image file."}
            )

        try:
            logger.info(f"Analyzing image: {file.filename}")
            
            analysis_result = face_analyzer.analyze_image(contents)
            
            logger.info("Face analysis completed successfully")
            
        except ValueError as e:
            logger.error(f"Validation error during analysis: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"detail": str(e)}
            )
        except Exception as e:
            logger.error(f"Error during face analysis: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error during face analysis: {str(e)}"}
            )

        amazon_scraper.recommended_products.clear()
        
        recommendations = []
        search_categories = []
        
        if analysis_result['skin_tone']['main_tone'] != "unknown":
            search_categories.append({
                "category": f"Skincare for {analysis_result['skin_tone']['main_tone'].title()} Skin",
                "keywords": [analysis_result['skin_tone']['main_tone'], "skincare"]
            })

        for concern, values in analysis_result['concerns'].items():
            if values:
                search_categories.append({
                    "category": f"{concern.title()} Treatment",
                    "keywords": values + ["skincare"]
                })

        for texture, values in analysis_result['texture'].items():
            if values:
                search_categories.append({
                    "category": f"{texture.title()} Treatment",
                    "keywords": values + ["skincare"]
                })

        logger.info(f"Generated search categories: {search_categories}")

        for category in search_categories:
            try:
                logger.info(f"Searching products for category: {category['category']}")
                products = amazon_scraper.search_products(category["keywords"])
                if products:
                    product_dicts = [product.dict() for product in products]
                    recommendations.append({
                        "category": category["category"],
                        "keywords": category["keywords"],
                        "products": product_dicts
                    })
                    logger.info(f"Found {len(products)} products for {category['category']}")
                else:
                    logger.warning(f"No products found for {category['category']}")
                    
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Error searching products for {category['category']}: {str(e)}")
                continue

        response_data = {
            "analysis": analysis_result,
            "recommendations": recommendations
        }
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "face_analyzer": "ok",
            "amazon_scraper": "ok"
        }
    }

