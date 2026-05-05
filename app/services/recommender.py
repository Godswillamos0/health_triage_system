import math
import pandas as pd
import os

class RecommenderService:
    def __init__(self, data_path="app/data/hospitals.csv"):
        self.data_path = data_path
        self.facilities_df = pd.DataFrame()
        self.load_facilities()

    def load_facilities(self):
        """Loads hospital and facility data from the CSV database."""
        if os.path.exists(self.data_path):
            self.facilities_df = pd.read_csv(self.data_path)
        else:
            # Fallback schema if the file hasn't been created yet
            self.facilities_df = pd.DataFrame(columns=[
                "id", "name", "latitude", "longitude", "services", "beds_available", "specialty"
            ])

    def calculate_haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates the great-circle distance between two points 
        on the Earth (in kilometers).
        """
        R = 6371.0  # Radius of Earth in kilometers

        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)

        a = (math.sin(d_lat / 2.0) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(d_lon / 2.0) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def rank_facilities(self, user_lat: float, user_lon: float, required_specialty: str, limit: int = 3):
        """
        Calculates distances using the Haversine formula and ranks facilities
        based on distance and available capacity.
        """
        if self.facilities_df.empty:
            return []

        df = self.facilities_df.copy()

        # Calculate distance for every facility
        df['distance_km'] = df.apply(
            lambda row: self.calculate_haversine(user_lat, user_lon, row['latitude'], row['longitude']),
            axis=1
        )

        # Multi-Criteria Decision Making (MCDM) scoring function
        # Prioritizes closer facilities with available beds and matching specialties
        df['score'] = df.apply(
            lambda row: (1.0 / (row['distance_km'] + 1.0)) + 
                        (1.0 if str(row['specialty']).strip().lower() == required_specialty.strip().lower() else 0.0) + 
                        (row['beds_available'] * 0.1),
            axis=1
        )

        # Sort and return the top facilities
        ranked_df = df.sort_values(by='score', ascending=False).head(limit)
        
        return ranked_df.to_dict(orient='records')

recommender_service = RecommenderService()