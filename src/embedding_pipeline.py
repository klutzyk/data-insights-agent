# embedding_pipeline.py - Complete pipeline for creating embeddings and vector store
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path

from .embeddings import EmbeddingGenerator, create_text_summaries
from .vector_store import VectorStore
from .data_loader import load_df, summarize_df

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    """
    Complete pipeline for creating embeddings from datasets and building vector stores.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_type: str = "flat",
                 mapping_type: str = "json"):
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name (str): Sentence transformer model name
            index_type (str): FAISS index type ("flat" or "ivf")
            mapping_type (str): Mapping store type ("sqlite" or "json")
        """
        self.model_name = model_name
        self.index_type = index_type
        self.mapping_type = mapping_type
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.vector_store = None
        self.dimension = self.embedding_generator.get_embedding_dimension()
        
        logger.info(f"EmbeddingPipeline initialized with {model_name} model")
    
    def create_vector_store_from_dataframe(self, 
                                         df: pd.DataFrame,
                                         text_columns: List[str] = None,
                                         batch_size: int = 32) -> VectorStore:
        """
        Create a vector store from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_columns (List[str]): Columns to include in text summaries
            batch_size (int): Batch size for embedding generation
            
        Returns:
            VectorStore: Created vector store
        """
        logger.info(f"Starting vector store creation from DataFrame with {len(df)} rows")
        
        # Create text summaries for each record
        logger.info("Creating text summaries...")
        text_summaries = create_text_summaries(df, text_columns)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings_batch(
            text_summaries, batch_size=batch_size
        )
        
        # Create metadata for each vector
        metadata_list = []
        for idx, row in df.iterrows():
            metadata = {
                'record_id': idx,
                'text_summary': text_summaries[idx],
                'original_data': row.to_dict()
            }
            metadata_list.append(metadata)
        
        # Create vector store
        logger.info("Creating vector store...")
        self.vector_store = VectorStore(
            dimension=self.dimension,
            index_type=self.index_type,
            mapping_type=self.mapping_type
        )
        
        # Add vectors to store
        vector_ids = self.vector_store.add_vectors(embeddings, metadata_list)
        
        logger.info(f"Vector store created with {len(vector_ids)} vectors")
        return self.vector_store
    
    
    def search_similar_records(self, 
                             query_text: str, 
                             k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar records using text query.
        
        Args:
            query_text (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[Dict]: Search results with metadata
        """
        if self.vector_store is None:
            raise ValueError("No vector store created. Call create_vector_store_from_dataframe() first.")
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        logger.info(f"Found {len(results)} similar records for query: '{query_text[:50]}...'")
        return results
    
    def save_vector_store(self, 
                         index_path: str, 
                         mapping_path: str = None):
        """
        Save the vector store to disk.
        
        Args:
            index_path (str): Path to save FAISS index
            mapping_path (str): Path to save mapping store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save.")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if mapping_path:
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        
        self.vector_store.save(index_path, mapping_path)
        logger.info(f"Vector store saved to {index_path}")
    
    def load_vector_store(self, 
                         index_path: str, 
                         mapping_path: str = None):
        """
        Load a vector store from disk.
        
        Args:
            index_path (str): Path to load FAISS index from
            mapping_path (str): Path to load mapping store from
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Create new vector store
        self.vector_store = VectorStore(
            dimension=self.dimension,
            index_type=self.index_type,
            mapping_type=self.mapping_type
        )
        
        # Load from disk
        self.vector_store.load(index_path, mapping_path)
        logger.info(f"Vector store loaded from {index_path}")
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current vector store.
        
        Returns:
            Dict: Statistics dictionary
        """
        if self.vector_store is None:
            return {"status": "No vector store created"}
        
        stats = self.vector_store.get_stats()
        stats.update({
            "model_name": self.model_name,
            "embedding_dimension": self.dimension
        })
        return stats
    


def create_embedding_pipeline_from_csv(csv_path: str,
                                     output_dir: str = "data/processed",
                                     text_columns: List[str] = None,
                                     model_name: str = "all-MiniLM-L6-v2",
                                     batch_size: int = 32) -> EmbeddingPipeline:
    """
    Convenience function to create a complete embedding pipeline from CSV.
    
    Args:
        csv_path (str): Path to CSV file
        output_dir (str): Directory to save vector store files
        text_columns (List[str]): Columns to include in text summaries
        model_name (str): Sentence transformer model name
        batch_size (int): Batch size for embedding generation
        
    Returns:
        EmbeddingPipeline: Created pipeline with vector store
    """
    # Create pipeline
    pipeline = EmbeddingPipeline(model_name=model_name)
    
    # Create vector store from CSV
    vector_store = pipeline.create_vector_store_from_csv(
        csv_path, text_columns, batch_size
    )
    
    # Save vector store
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "vector_index.faiss")
    mapping_path = os.path.join(output_dir, "vector_mappings.json")
    
    pipeline.save_vector_store(index_path, mapping_path)
    
    logger.info(f"Complete pipeline created and saved to {output_dir}")
    return pipeline

if __name__ == "__main__":
    # Test the pipeline
    pipeline = EmbeddingPipeline()
    
    # Create some test data
    test_data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
        'occupation': ['Engineer', 'Doctor', 'Teacher', 'Artist', 'Lawyer']
    }
    df = pd.DataFrame(test_data)
    
    # Create vector store
    vector_store = pipeline.create_vector_store_from_dataframe(df)
    
    # Search for similar records
    results = pipeline.search_similar_records("young engineer in big city", k=3)
    
    print("Search results:")
    for result in results:
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Metadata: {result['metadata']['original_data']}")
        print("---")
    
    # Get stats
    stats = pipeline.get_vector_store_stats()
    print(f"Vector store stats: {stats}")
