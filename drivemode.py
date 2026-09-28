# drivemode.py - Complete ML-based drive mode with severity classification
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

class MLDriveModeRecommender:
    """
    Machine Learning-based drive mode recommendation with severity classification.
    Features: pothole count, confidence, rate of change, time patterns, road conditions.
    """
    
    def __init__(self, model_path='drive_mode_model.pkl'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None
        
        # Mode mappings
        self.modes = [
            "Smooth Drive",
            "Highway Mode", 
            "Comfort Mode",
            "Caution Mode",
            "Rough Road Mode",
            "Danger! Slow Down"
        ]
        
        # ✅ Severity classifications
        self.severity_levels = {
            0: {
                "level": "Minimal",
                "description": "Road is in excellent condition",
                "score": 0,
                "color": "#00FF00",
                "color_bgr": (0, 255, 0),
                "action": "Continue at normal speed",
                "risk": "Very Low"
            },
            1: {
                "level": "Very Low",
                "description": "Minor surface imperfections",
                "score": 1,
                "color": "#7FFF00",
                "color_bgr": (0, 255, 127),
                "action": "Maintain current speed",
                "risk": "Low"
            },
            2: {
                "level": "Low",
                "description": "Some potholes present, manageable",
                "score": 2,
                "color": "#FFFF00",
                "color_bgr": (0, 255, 255),
                "action": "Slight speed reduction recommended",
                "risk": "Low-Medium"
            },
            3: {
                "level": "Moderate",
                "description": "Multiple hazards detected",
                "score": 3,
                "color": "#FFA500",
                "color_bgr": (0, 165, 255),
                "action": "Reduce speed, increase caution",
                "risk": "Medium"
            },
            4: {
                "level": "High",
                "description": "Significant road damage present",
                "score": 4,
                "color": "#FF4500",
                "color_bgr": (0, 69, 255),
                "action": "Slow down significantly",
                "risk": "High"
            },
            5: {
                "level": "Critical",
                "description": "Dangerous road conditions",
                "score": 5,
                "color": "#FF0000",
                "color_bgr": (0, 0, 255),
                "action": "SLOW DOWN IMMEDIATELY",
                "risk": "Very High"
            }
        }
        
        # History tracking
        self.recent_counts = []
        self.recent_confidences = []
        self.recent_timestamps = []
        self.last_update_time = time.time()
        self.last_mode = "Smooth Drive"
        self.mode_hold_until = 0
        
        # Load or initialize model
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create and train a new one"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                print("✅ Loaded existing ML model")
                return
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
        
        # Create and train new model with synthetic data
        print("🔧 Training new ML model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self._train_initial_model()
    
    def _train_initial_model(self):
        """Train model with synthetic driving data"""
        # Generate synthetic training data
        X_train, y_train = self._generate_training_data()
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        
        # Save model
        self._save_model()
        print(f"✅ Model trained with {len(X_train)} samples")
    
    def _generate_training_data(self):
        """Generate synthetic training data based on driving scenarios"""
        X = []
        y = []
        
        # Smooth Drive scenarios (0)
        for _ in range(200):
            X.append([
                np.random.uniform(0, 0.5),      # avg_potholes
                np.random.uniform(85, 100),     # confidence
                np.random.uniform(-0.1, 0.1),   # rate_of_change
                np.random.uniform(0, 1),        # variance
                0                                # recent_danger
            ])
            y.append(0)
        
        # Highway Mode scenarios (1)
        for _ in range(150):
            X.append([
                0,                               # avg_potholes
                np.random.uniform(90, 100),      # confidence
                0,                               # rate_of_change
                0,                               # variance
                0                                # recent_danger
            ])
            y.append(1)
        
        # Comfort Mode scenarios (2)
        for _ in range(200):
            X.append([
                np.random.uniform(0.5, 2),       # avg_potholes
                np.random.uniform(70, 90),       # confidence
                np.random.uniform(-0.2, 0.2),    # rate_of_change
                np.random.uniform(0.2, 0.8),     # variance
                0                                # recent_danger
            ])
            y.append(2)
        
        # Caution Mode scenarios (3)
        for _ in range(200):
            X.append([
                np.random.uniform(2, 5),         # avg_potholes
                np.random.uniform(60, 85),       # confidence
                np.random.uniform(0, 0.5),       # rate_of_change
                np.random.uniform(0.5, 1.5),     # variance
                0                                # recent_danger
            ])
            y.append(3)
        
        # Rough Road Mode scenarios (4)
        for _ in range(200):
            X.append([
                np.random.uniform(5, 8),         # avg_potholes
                np.random.uniform(65, 85),       # confidence
                np.random.uniform(0.2, 0.8),     # rate_of_change
                np.random.uniform(1, 2),         # variance
                np.random.uniform(0, 1)          # recent_danger
            ])
            y.append(4)
        
        # Danger scenarios (5)
        for _ in range(150):
            X.append([
                np.random.uniform(8, 15),        # avg_potholes
                np.random.uniform(70, 95),       # confidence
                np.random.uniform(0.5, 2),       # rate_of_change
                np.random.uniform(1.5, 3),       # variance
                1                                # recent_danger
            ])
            y.append(5)
        
        return np.array(X), np.array(y)
    
    def _extract_features(self, potholes, confidence):
        """Extract features from current and historical data"""
        current_time = time.time()
        
        # Normalize confidence
        if confidence <= 1:
            confidence *= 100
        
        # Update history every second
        if current_time - self.last_update_time > 1.0:
            if len(self.recent_counts) >= 10:
                self.recent_counts.pop(0)
                self.recent_confidences.pop(0)
                self.recent_timestamps.pop(0)
            
            self.recent_counts.append(potholes)
            self.recent_confidences.append(confidence)
            self.recent_timestamps.append(current_time)
            self.last_update_time = current_time
        
        # Calculate features
        if len(self.recent_counts) > 0:
            avg_potholes = np.mean(self.recent_counts)
            variance = np.var(self.recent_counts) if len(self.recent_counts) > 1 else 0
            
            # Rate of change (potholes per second)
            if len(self.recent_counts) >= 2:
                time_diff = self.recent_timestamps[-1] - self.recent_timestamps[0]
                count_diff = self.recent_counts[-1] - self.recent_counts[0]
                rate_of_change = count_diff / max(time_diff, 1)
            else:
                rate_of_change = 0
            
            # Recent danger flag (any high pothole count in last 5 seconds)
            recent_danger = 1 if any(c > 6 for c in self.recent_counts[-5:]) else 0
        else:
            avg_potholes = float(potholes)
            variance = 0
            rate_of_change = 0
            recent_danger = 1 if potholes > 6 else 0
        
        return np.array([[
            avg_potholes,
            confidence,
            rate_of_change,
            variance,
            recent_danger
        ]])
    
    def _calculate_severity(self, mode_idx, potholes, confidence, avg_potholes):
        """
        ✅ Calculate severity based on mode, pothole count, and conditions
        Returns comprehensive severity data
        """
        # Map mode to base severity
        mode_severity_map = {
            0: 0,  # Smooth Drive -> Minimal
            1: 0,  # Highway Mode -> Minimal
            2: 2,  # Comfort Mode -> Low
            3: 3,  # Caution Mode -> Moderate
            4: 4,  # Rough Road Mode -> High
            5: 5   # Danger -> Critical
        }
        
        base_severity = mode_severity_map.get(mode_idx, 2)
        
        # Adjust severity based on actual conditions
        if avg_potholes > 10:
            severity_idx = 5  # Critical
        elif avg_potholes > 7:
            severity_idx = 4  # High
        elif avg_potholes > 4:
            severity_idx = 3  # Moderate
        elif avg_potholes > 1.5:
            severity_idx = 2  # Low
        elif avg_potholes > 0.5:
            severity_idx = 1  # Very Low
        else:
            severity_idx = 0  # Minimal
        
        # Take the higher severity between mode-based and pothole-based
        final_severity = max(base_severity, severity_idx)
        
        # Get severity data
        severity_data = self.severity_levels[final_severity].copy()
        
        # Add additional metrics
        severity_data["pothole_count"] = int(potholes)
        severity_data["avg_potholes"] = round(avg_potholes, 2)
        severity_data["confidence"] = round(confidence, 1)
        
        # Calculate damage estimate (0-100%)
        damage_percent = min(100, (avg_potholes / 15) * 100)
        severity_data["damage_estimate"] = round(damage_percent, 1)
        
        return severity_data
    
    def recommend_mode(self, potholes, confidence, fps=30):
        """
        ML-based drive mode recommendation with severity classification
        
        Args:
            potholes: Current number of potholes detected
            confidence: Detection confidence (0-1 or 0-100)
            fps: Frames per second (for future use)
        
        Returns:
            dict with mode, confidence, severity, and additional insights
        """
        current_time = time.time()
        
        # Extract features
        features = self._extract_features(potholes, confidence)
        
        # Normalize confidence for display
        if confidence <= 1:
            confidence *= 100
        
        # Make prediction
        features_scaled = self.scaler.transform(features)
        mode_idx = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        mode = self.modes[mode_idx]
        mode_confidence = probabilities[mode_idx] * 100
        
        # Calculate average potholes for display
        avg_potholes = np.mean(self.recent_counts) if self.recent_counts else float(potholes)
        
        # ✅ Calculate severity
        severity_data = self._calculate_severity(mode_idx, potholes, confidence, avg_potholes)
        
        # Low confidence adjustment
        if confidence < 60 and mode != "Smooth Drive":
            mode += " (Low Confidence)"
        
        # Smart transition logic (prevent flickering)
        caution_modes = ["Caution Mode", "Rough Road Mode", "Danger! Slow Down"]
        if any(m in self.last_mode for m in caution_modes):
            if self.mode_hold_until == 0:
                self.mode_hold_until = current_time + 3.0
            
            if current_time < self.mode_hold_until and mode == "Smooth Drive":
                mode = self.last_mode
                mode_confidence = probabilities[self.modes.index(self.last_mode.split(" (")[0])] * 100
            elif current_time >= self.mode_hold_until:
                self.mode_hold_until = 0
        
        self.last_mode = mode
        
        return {
            "mode": mode,
            "avg_potholes": round(avg_potholes, 2),
            "mode_confidence": round(mode_confidence, 1),
            "severity": severity_data,  # ✅ Full severity classification
            "top_3_modes": [
                (self.modes[i], round(probabilities[i] * 100, 1))
                for i in np.argsort(probabilities)[-3:][::-1]
            ]
        }
    
    def _save_model(self):
        """Save trained model and scaler"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler
                }, f)
            print(f"💾 Model saved to {self.model_path}")
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
    
    def get_feature_importance(self):
        """Get feature importance for model interpretation"""
        feature_names = [
            'avg_potholes',
            'confidence', 
            'rate_of_change',
            'variance',
            'recent_danger'
        ]
        
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))


# Create global instance
_recommender = None

def get_recommender():
    """Get or create ML recommender instance"""
    global _recommender
    if _recommender is None:
        _recommender = MLDriveModeRecommender()
    return _recommender

def recommend_mode(potholes, confidence, fps=30):
    """
    Main interface function - compatible with original API
    
    Returns dict with:
        - mode: Drive mode recommendation
        - avg_potholes: Average pothole count over time
        - mode_confidence: ML model confidence
        - severity: Complete severity classification data
        - top_3_modes: Top 3 predicted modes with probabilities
    """
    recommender = get_recommender()
    return recommender.recommend_mode(potholes, confidence, fps)


# Example usage and testing
if __name__ == "__main__":
    recommender = MLDriveModeRecommender()
    
    print("\n🚗 Simulating driving scenarios with severity...\n")
    
    scenarios = [
        (0, 95, "Clean highway"),
        (1, 85, "Smooth city road"),
        (3, 80, "Minor potholes"),
        (6, 75, "Rough patch"),
        (10, 85, "Dangerous road"),
        (2, 70, "Improving conditions"),
        (0, 90, "Back to smooth")
    ]
    
    for potholes, conf, desc in scenarios:
        result = recommender.recommend_mode(potholes, conf)
        severity = result['severity']
        
        print(f"📍 {desc}: {potholes} potholes, {conf}% confidence")
        print(f"  → Mode: {result['mode']}")
        print(f"  → ML Confidence: {result['mode_confidence']}%")
        print(f"  → Avg Potholes: {result['avg_potholes']}")
        print(f"  ⚠️ SEVERITY: {severity['level']} ({severity['description']})")
        print(f"     • Risk Level: {severity['risk']}")
        print(f"     • Damage Estimate: {severity['damage_estimate']}%")
        print(f"     • Recommended Action: {severity['action']}\n")
        time.sleep(1.5)
    
    print("\n📊 Feature Importance:")
    for feature, importance in recommender.get_feature_importance().items():
        print(f"  {feature}: {importance:.3f}")