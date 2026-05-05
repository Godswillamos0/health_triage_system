import re
import string

class NLPEngine:
    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """
        Cleans and normalizes symptom text by converting to lowercase 
        and removing punctuation.
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_symptoms(self, text: str) -> list[str]:
        """
        Extracts and tokenizes individual symptoms from the normalized text.
        """
        normalized_text = self.normalize_text(text)
        
        # Split by commas or conjunctions to extract symptom components
        tokens = re.split(r',|\band\b', normalized_text)
        
        return [token.strip() for token in tokens if token.strip()]