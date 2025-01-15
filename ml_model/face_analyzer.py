import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import skew, kurtosis

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        # Define regions of interest for different analyses
        self.forehead_points = [10, 108, 151, 337, 299]
        self.cheek_points = [123, 147, 192, 213, 123, 147, 192, 213]
        self.under_eye_points = [157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155]

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        target_width, target_height = target_size
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a black canvas and embed the resized image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        return canvas

    def _create_mask_from_landmarks(self, image: np.ndarray, landmarks, points: List[int]) -> np.ndarray:
        """Create a mask for specific facial regions using landmarks"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        polygon_points = []
        for point in points:
            x = int(landmarks.landmark[point].x * width)
            y = int(landmarks.landmark[point].y * height)
            polygon_points.append([x, y])
            
        if len(polygon_points) > 2:
            polygon_points = np.array([polygon_points], dtype=np.int32)
            cv2.fillPoly(mask, polygon_points, 255)
            
        return mask

    def _analyze_skin_tone(self, image: np.ndarray, landmarks) -> Dict[str, any]:
        # Create mask for cheek region
        cheek_mask = self._create_mask_from_landmarks(image, landmarks, self.cheek_points)
        
        # Convert to LAB color space for better color analysis
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract skin tone from masked region
        masked_lab = cv2.bitwise_and(lab_image, lab_image, mask=cheek_mask)
        valid_pixels = masked_lab[cheek_mask > 0]
        
        if len(valid_pixels) > 0:
            avg_lab = np.mean(valid_pixels, axis=0)
            
            # L channel represents lightness
            lightness = avg_lab[0]
            
            # Classify tone
            if lightness > 85:
                tone = "fair"
                undertone = "cool" if avg_lab[1] < 128 else "warm"
            elif lightness > 70:
                tone = "medium"
                undertone = "neutral" if 127 < avg_lab[1] < 129 else "warm"
            else:
                tone = "dark"
                undertone = "warm" if avg_lab[1] > 128 else "neutral"
                
            return {
                "main_tone": tone,
                "undertone": undertone,
                "lightness_value": float(lightness)
            }
        return {"main_tone": "unknown", "undertone": "unknown", "lightness_value": 0.0}

    def _analyze_skin_texture(self, image: np.ndarray, landmarks) -> Dict[str, List[str]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        
        # Create masks for different facial regions
        forehead_mask = self._create_mask_from_landmarks(image, landmarks, self.forehead_points)
        cheek_mask = self._create_mask_from_landmarks(image, landmarks, self.cheek_points)
        
        texture_features = {
            "forehead": [],
            "cheeks": [],
            "overall": []
        }
        
        # Analyze each region
        for region, mask in [("forehead", forehead_mask), ("cheeks", cheek_mask)]:
            # Apply mask
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            
            if np.sum(mask) > 0:
                # Calculate texture metrics
                roi_valid = roi[mask > 0]
                
                # Local Binary Pattern for texture analysis
                lbp = self._local_binary_pattern(roi)
                lbp_masked = cv2.bitwise_and(lbp, lbp, mask=mask)
                
                # Calculate statistics
                std_dev = np.std(roi_valid)
                entropy = self._calculate_entropy(roi_valid)
                roughness = self._calculate_roughness(lbp_masked, mask)
                
                # Classify textures based on metrics
                if std_dev > 25:
                    texture_features[region].append("uneven")
                else:
                    texture_features[region].append("even")
                    
                if entropy > 5:
                    texture_features[region].append("textured")
                else:
                    texture_features[region].append("smooth")
                    
                if roughness > 0.5:
                    texture_features[region].append("rough")
                else:
                    texture_features[region].append("refined")
        
        # Overall texture analysis
        texture_features["overall"] = self._combine_texture_features(texture_features)
        return texture_features

    def _detect_skin_concerns(self, image: np.ndarray, landmarks) -> Dict[str, List[str]]:
        concerns = {
            "forehead": [],
            "cheeks": [],
            "under_eyes": [],
            "overall": []
        }
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Analyze different regions
        regions = {
            "forehead": (self.forehead_points, self._analyze_forehead),
            "cheeks": (self.cheek_points, self._analyze_cheeks),
            "under_eyes": (self.under_eye_points, self._analyze_under_eyes)
        }
        
        for region, (points, analysis_func) in regions.items():
            mask = self._create_mask_from_landmarks(image, landmarks, points)
            concerns[region] = analysis_func(image, gray, ycrcb, mask)
        
        # Combine concerns for overall analysis
        concerns["overall"] = self._combine_concerns(concerns)
        return concerns

    def _analyze_forehead(self, image: np.ndarray, gray: np.ndarray, ycrcb: np.ndarray, mask: np.ndarray) -> List[str]:
        concerns = []
        
        # Analyze wrinkles
        edges = cv2.Canny(gray, 50, 150)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        if np.mean(masked_edges) > 10:
            concerns.append("fine_lines")
        
        # Analyze oil/shine
        y_channel = ycrcb[:,:,0]
        masked_y = cv2.bitwise_and(y_channel, y_channel, mask=mask)
        if np.mean(masked_y[mask > 0]) > 200:
            concerns.append("oily")
        elif np.mean(masked_y[mask > 0]) < 130:
            concerns.append("dry")
            
        return concerns

    def _analyze_cheeks(self, image: np.ndarray, gray: np.ndarray, ycrcb: np.ndarray, mask: np.ndarray) -> List[str]:
        concerns = []
        
        # Analyze redness
        b, g, r = cv2.split(image)
        redness = r - g
        masked_redness = cv2.bitwise_and(redness, redness, mask=mask)
        if np.mean(masked_redness[mask > 0]) > 30:
            concerns.append("redness")
            
        # Analyze pores
        local_std = cv2.blur(gray, (5,5)) - cv2.blur(gray, (15,15))
        masked_std = cv2.bitwise_and(local_std, local_std, mask=mask)
        if np.mean(masked_std[mask > 0]) > 15:
            concerns.append("visible_pores")
            
        return concerns

    def _analyze_under_eyes(self, image: np.ndarray, gray: np.ndarray, ycrcb: np.ndarray, mask: np.ndarray) -> List[str]:
        concerns = []
        
        # Analyze dark circles
        y_channel = ycrcb[:,:,0]
        masked_y = cv2.bitwise_and(y_channel, y_channel, mask=mask)
        if np.mean(masked_y[mask > 0]) < 140:
            concerns.append("dark_circles")
            
        # Analyze puffiness
        local_var = cv2.blur(gray, (15,15)) - cv2.blur(gray, (5,5))
        masked_var = cv2.bitwise_and(local_var, local_var, mask=mask)
        if np.mean(masked_var[mask > 0]) > 10:
            concerns.append("puffiness")
            
        return concerns

    def _local_binary_pattern(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern for texture analysis"""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i,j]
                code = 0
                code |= (image[i-1,j-1] > center) << 7
                code |= (image[i-1,j] > center) << 6
                code |= (image[i-1,j+1] > center) << 5
                code |= (image[i,j+1] > center) << 4
                code |= (image[i+1,j+1] > center) << 3
                code |= (image[i+1,j] > center) << 2
                code |= (image[i+1,j-1] > center) << 1
                code |= (image[i,j-1] > center) << 0
                lbp[i,j] = code
        return lbp

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy as a measure of texture complexity"""
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        hist = hist.ravel() / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-7))

    def _calculate_roughness(self, lbp: np.ndarray, mask: np.ndarray) -> float:
        """Calculate surface roughness using LBP"""
        valid_lbp = lbp[mask > 0]
        if len(valid_lbp) > 0:
            return np.var(valid_lbp) / 10000.0
        return 0.0

    def _combine_texture_features(self, texture_features: Dict[str, List[str]]) -> List[str]:
        """Combine texture features from different regions"""
        combined = []
        for feature in ["uneven", "textured", "rough"]:
            count = sum(1 for region in ["forehead", "cheeks"] 
                       if feature in texture_features[region])
            if count >= 1:
                combined.append(feature)
        return combined

    def _combine_concerns(self, concerns: Dict[str, List[str]]) -> List[str]:
        """Combine concerns from different regions"""
        all_concerns = []
        region_concerns = [c for region in ["forehead", "cheeks", "under_eyes"]
                         for c in concerns[region]]
        
        # Add concerns that appear in multiple regions
        from collections import Counter
        concern_counts = Counter(region_concerns)
        for concern, count in concern_counts.items():
            if count >= 1:  # Adjust threshold as needed
                all_concerns.append(concern)
                
        return list(set(all_concerns))

    def analyze_image(self, image_bytes: bytes) -> Dict[str, any]:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image")

        # Get the first face detected
        face_landmarks = results.multi_face_landmarks[0]

        # Perform comprehensive analysis
        tone_analysis = self._analyze_skin_tone(image, face_landmarks)
        texture_analysis = self._analyze_skin_texture(image, face_landmarks)
        concerns_analysis = self._detect_skin_concerns(image, face_landmarks)

        # Generate product keywords
        keywords = self._generate_product_keywords(tone_analysis, texture_analysis, concerns_analysis)

        return {
            "skin_tone": tone_analysis,
            "texture": texture_analysis,
            "concerns": concerns_analysis,
            "keywords": keywords,
            "confidence_score": 0.85
        }

    def _generate_product_keywords(self, tone, texture, concerns) -> List[str]:
        keywords = set()
        
        # Add keywords based on skin tone
        tone_keywords = {
            "fair": ["brightening", "sun protection", "spf"],
            "medium": ["balancing", "hydrating"],
            "dark": ["evening", "hydrating", "dark spot correction"]
        }
        if tone["main_tone"] in tone_keywords:
            keywords.update(tone_keywords[tone["main_tone"]])
            
        # Add undertone specific keywords
        undertone_keywords = {
            "cool": ["neutral ph"],
            "warm": ["barrier repair"],
            "neutral": ["gentle"]
        }
        if tone["undertone"] in undertone_keywords:
            keywords.update(undertone_keywords[tone["undertone"]])

        # Add texture keywords
        texture_keywords = {
            "uneven": ["smoothing", "retexturizing"],
            "textured": ["exfoliating", "resurfacing"],
            "rough": ["smoothing", "gentle exfoliation"],
            "refined": ["maintaining", "protective"]
        }
        for feature in texture["overall"]:
            if feature in texture_keywords:
                keywords.update(texture_keywords[feature])

        # Add concerns keywords
        concern_keywords = {
            "fine_lines": ["anti-aging", "retinol", "peptides"],
            "oily": ["oil-control", "non-comedogenic", "mattifying"],
            "dry": ["hydrating", "moisturizing", "hyaluronic acid"],
            "redness": ["soothing", "calming", "anti-inflammatory"],
            "visible_pores": ["pore-minimizing", "clay"],
            "dark_circles": ["brightening", "caffeine", "vitamin k"],
            "puffiness": ["de-puffing", "cooling", "drainage"]
        }
        for concern in concerns["overall"]:
            if concern in concern_keywords:
                keywords.update(concern_keywords[concern])

        return list(keywords)

