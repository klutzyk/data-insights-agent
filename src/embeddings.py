# embeddings.py - generate embeddings for text units
import os
# Avoid importing torchvision via transformers for text-only models
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    A class to generate embeddings for text data using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Embedding vector
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call _load_model() first.")
        
        try:
            # Clean and prepare text
            clean_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(clean_text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call _load_model() first.")
        
        try:
            # Clean and prepare texts
            clean_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings in batches
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(
                clean_texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        
        # Handle empty or very short texts
        if len(text) < 2:
            text = "empty text"
        
        # Truncate very long texts (due to sentence-transformer limits)
        max_length = 512 
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            int: Embedding dimension
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


def create_text_summaries(df, text_columns: List[str] = None) -> List[str]:
    """
    Create text summaries from DataFrame records.
    
    Args:
        df: pandas DataFrame
        text_columns: List of column names to include in summary
        
    Returns:
        List[str]: List of text summaries for each record
    """
    import pandas as pd
    
    summaries = []
    
    # If no text columns specified, use all object columns
    if text_columns is None:
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    for idx, row in df.iterrows():
        summary_parts = []
        
        # Add record index
        summary_parts.append(f"Record {idx}:")
        
        # Add specified columns
        for col in text_columns:
            if col in df.columns:
                value = row[col]
                if pd.notna(value):
                    summary_parts.append(f"{col}: {value}")
        
        # Add numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_summary = []
            for col in numeric_cols:
                if pd.notna(row[col]):
                    numeric_summary.append(f"{col}={row[col]}")
            if numeric_summary:
                summary_parts.append(f"Numeric: {', '.join(numeric_summary)}")
        
        summary = " | ".join(summary_parts)
        summaries.append(summary)
    
    return summaries


if __name__ == "__main__":
    # Test the embedding generator
    generator = EmbeddingGenerator()
    
    # Test single embedding
    test_text = "This is a test sentence for embedding generation."
    embedding = generator.generate_embedding(test_text)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Test batch embeddings
    test_texts = [
        "First test sentence",
        "Second test sentence", 
        "Third test sentence"
    ]
    embeddings = generator.generate_embeddings_batch(test_texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = generator.similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two embeddings: {sim:.4f}")
