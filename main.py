from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import json
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
    allow_origins=["*"],
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

class ProductRecommendation(BaseModel):
    category: str
    keywords: List[str]
    products: List[ProductDetails]

class FullResponse(BaseModel):
    analysis: dict
    recommendations: List[ProductRecommendation]


class ProductDatabase:
    def __init__(self):
        self.load_database()
        self.recommended_products = set()  # Track all recommended products by URL

    def load_database(self):
        """Load product database from JSON file"""
        try:
            with open('product_database.json', 'r') as f:
                self.database = json.load(f)
            logger.info("Successfully loaded product database")
        except Exception as e:
            logger.error(f"Error loading product database: {str(e)}")
            self.database = []

    def reset_recommendations(self):
        """Reset the set of recommended products"""
        self.recommended_products = set()

    def find_matching_products(self, category: str, keywords: List[str], num_products: int = 5) -> List[ProductDetails]:
        """Find unique products matching category and keywords"""
        try:
            # Find matching category in database
            category_lower = category.lower()
            matching_category = None
            
            # Map common variations of category names
            category_mappings = {
                "skincare for fair skin": ["skincare for fair skin", "fair skin"],
                "skincare for medium skin": ["skincare for medium skin", "medium skin"],
                "skincare for dark skin": ["skincare for dark skin", "dark skin"],
                "acne treatment": ["acne treatment", "acne"],
                "aging treatment": ["aging treatment", "anti aging", "aging"],
                "dry skin treatment": ["dry skin treatment", "dry skin"],
                "oily skin treatment": ["oily skin treatment", "oily skin"],
                "sensitive skin treatment": ["sensitive skin treatment", "sensitive skin"],
                "combination skin treatment": ["combination skin treatment", "combination skin"],
                "texture treatment": ["texture treatment", "texture", "uneven texture"]
            }
            
            # Find the matching category using the mappings
            for cat in self.database:
                cat_name = cat["category"].lower()
                for mapped_category, variants in category_mappings.items():
                    if category_lower in variants or cat_name in variants:
                        matching_category = cat
                        break
                if matching_category:
                    break

            if not matching_category:
                logger.warning(f"No matching category found for: {category}")
                return []

            # Convert keywords to lowercase for case-insensitive matching
            keywords_lower = [k.lower() for k in keywords]

            # Score and filter products
            scored_products = []
            for product in matching_category["products"]:
                # Skip if product URL already recommended
                if product["product_url"] in self.recommended_products:
                    continue

                title_lower = product["title"].lower()
                score = sum(1 for keyword in keywords_lower if keyword in title_lower)
                
                # Include product if it matches keywords or as a potential backup
                scored_products.append((
                    score,
                    product.get("rating", 0) or 0,
                    product.get("reviews_count", 0) or 0,
                    product
                ))

            # Sort products by score, rating, and review count
            scored_products.sort(reverse=True, key=lambda x: (
                x[0],  # Keyword match score
                x[1],  # Rating
                x[2]   # Review count
            ))

            # Select top unique products
            selected_products = []
            for _, _, _, product in scored_products:
                if len(selected_products) >= num_products:
                    break
                    
                product_url = product["product_url"]
                if product_url not in self.recommended_products:
                    selected_products.append(ProductDetails(**product))
                    self.recommended_products.add(product_url)

            # If we don't have enough products, try to add more from remaining products
            if len(selected_products) < num_products:
                remaining_products = [
                    p[3] for p in scored_products 
                    if p[3]["product_url"] not in self.recommended_products
                ]
                
                for product in remaining_products:
                    if len(selected_products) >= num_products:
                        break
                    selected_products.append(ProductDetails(**product))
                    self.recommended_products.add(product["product_url"])

            return selected_products

        except Exception as e:
            logger.error(f"Error finding matching products: {str(e)}")
            return []

product_db = ProductDatabase()

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
                products = product_db.find_matching_products(
                    category["category"],
                    category["keywords"]
                )
                if products:
                    recommendations.append({
                        "category": category["category"],
                        "keywords": category["keywords"],
                        "products": [product.dict() for product in products]
                    })
                    logger.info(f"Found {len(products)} unique products for {category['category']}")
                else:
                    logger.warning(f"No unique products found for {category['category']}")

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
            "product_database": "ok"
        }
    }
