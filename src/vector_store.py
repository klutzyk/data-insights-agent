# vector_store.py - FAISS integration (index, search, save/load)
import faiss
import numpy as np
import json
import sqlite3
import pickle
from typing import List, Dict, Tuple, Optional, Any
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    A vector store implementation using FAISS for similarity search.
    Includes mapping between FAISS internal IDs and dataset records.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat", mapping_type: str = "sqlite"):
        """
        Initialize the vector store.
        
        Args:
            dimension (int): Dimension of the vectors
            index_type (str): Type of FAISS index ("flat" or "ivf")
            mapping_type (str): Type of mapping store ("sqlite" or "json")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.mapping_type = mapping_type
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Initialize mapping store
        self.mapping_store = self._create_mapping_store()
        
        # Track number of vectors
        self.vector_count = 0
        
        logger.info(f"VectorStore initialized with {dimension}D vectors, {index_type} index, {mapping_type} mapping")
    
    def _create_index(self):
        """Create FAISS index based on type."""
        if self.index_type == "flat":
            # Simple flat index 
            index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created FAISS IndexFlatL2")
        elif self.index_type == "ivf":
            # IVF index - better for large datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            logger.info("Created FAISS IndexIVFFlat")
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def _create_mapping_store(self):
        """Create mapping store for ID to record mapping."""
        if self.mapping_type == "sqlite":
            return SQLiteMappingStore()
        elif self.mapping_type == "json":
            return JSONMappingStore()
        else:
            raise ValueError(f"Unsupported mapping type: {self.mapping_type}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[int]:
        """
        Add vectors to the index with metadata.
        
        Args:
            vectors (np.ndarray): Array of vectors to add
            metadata (List[Dict]): List of metadata dictionaries for each vector
            
        Returns:
            List[int]: List of internal IDs assigned by FAISS
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match store dimension {self.dimension}")
        
        # Add vectors to FAISS index
        self.index.add(vectors.astype('float32'))
        
        # Get the IDs assigned by FAISS
        start_id = self.vector_count
        end_id = start_id + len(vectors)
        vector_ids = list(range(start_id, end_id))
        
        # Store metadata mapping
        for i, (vector_id, meta) in enumerate(zip(vector_ids, metadata)):
            self.mapping_store.add_mapping(vector_id, meta)
        
        self.vector_count += len(vectors)
        
        logger.info(f"Added {len(vectors)} vectors to index. Total vectors: {self.vector_count}")
        return vector_ids
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector (np.ndarray): Query vector
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of results with metadata and similarity scores
        """
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match store dimension {self.dimension}")
        
        # Ensure index is trained for IVF
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.warning("IVF index not trained. Training with existing vectors...")
            self.index.train(self.index.reconstruct_n(0, self.vector_count))
        
        # Search
        # reshaping 1D query vector to 2D array for FAISS search
        query_vector = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, min(k, self.vector_count))
        
        # Get metadata for results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            metadata = self.mapping_store.get_mapping(idx)
            if metadata:
                result = {
                    'id': idx,
                    'distance': float(distance),
                    'similarity': float(1 / (1 + distance)),  # Convert distance to similarity
                    'metadata': metadata
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} similar vectors")
        return results
    
    def get_vector(self, vector_id: int) -> Optional[np.ndarray]:
        """
        Retrieve a vector by its ID.
        
        Args:
            vector_id (int): Internal vector ID
            
        Returns:
            np.ndarray: Vector if found, None otherwise
        """
        if vector_id >= self.vector_count:
            return None
        
        try:
            vector = self.index.reconstruct(vector_id)
            return vector
        except:
            return None
    
    def get_metadata(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector ID.
        
        Args:
            vector_id (int): Internal vector ID
            
        Returns:
            Dict: Metadata if found, None otherwise
        """
        return self.mapping_store.get_mapping(vector_id)
    
    def save(self, index_path: str, mapping_path: str = None):
        """
        Save the vector store to disk.
        
        Args:
            index_path (str): Path to save FAISS index
            mapping_path (str): Path to save mapping store (optional)
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save mapping store
        if mapping_path:
            self.mapping_store.save(mapping_path)
            logger.info(f"Saved mapping store to {mapping_path}")
    
    def load(self, index_path: str, mapping_path: str = None):
        """
        Load the vector store from disk.
        
        Args:
            index_path (str): Path to load FAISS index from
            mapping_path (str): Path to load mapping store from (optional)
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.vector_count = self.index.ntotal
        logger.info(f"Loaded FAISS index from {index_path} with {self.vector_count} vectors")
        
        # Load mapping store
        if mapping_path and os.path.exists(mapping_path):
            self.mapping_store.load(mapping_path)
            logger.info(f"Loaded mapping store from {mapping_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics dictionary
        """
        return {
            'vector_count': self.vector_count,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'mapping_type': self.mapping_type,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }


class SQLiteMappingStore:
    """SQLite-based mapping store for vector IDs to metadata."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vector_mappings (
                id INTEGER PRIMARY KEY,
                metadata TEXT
            )
        ''')
        self.conn.commit()
    
    def add_mapping(self, vector_id: int, metadata: Dict[str, Any]):
        """Add a mapping between vector ID and metadata."""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO vector_mappings (id, metadata) VALUES (?, ?)',
            (vector_id, json.dumps(metadata))
        )
        self.conn.commit()
    
    def get_mapping(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT metadata FROM vector_mappings WHERE id = ?', (vector_id,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def save(self, path: str):
        """Save the database to a file."""
        if self.db_path == ":memory:":
            # Copy in-memory database to file
            backup = sqlite3.connect(path)
            self.conn.backup(backup)
            backup.close()
        else:
            # Copy current database to new path
            import shutil
            shutil.copy2(self.db_path, path)
    
    def load(self, path: str):
        """Load database from file."""
        if os.path.exists(path):
            self.db_path = path
            self.conn.close()
            self.conn = sqlite3.connect(path)


class JSONMappingStore:
    """JSON-based mapping store for vector IDs to metadata."""
    
    def __init__(self):
        self.mappings = {}
    
    def add_mapping(self, vector_id: int, metadata: Dict[str, Any]):
        """Add a mapping between vector ID and metadata."""
        self.mappings[vector_id] = metadata
    
    def get_mapping(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector ID."""
        return self.mappings.get(vector_id)
    
    def save(self, path: str):
        """Save mappings to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.mappings, f, indent=2)
    
    def load(self, path: str):
        """Load mappings from JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.mappings = json.load(f)


if __name__ == "__main__":
    # Test the vector store
    dimension = 384  # all-MiniLM-L6-v2 dimension
    
    # Create vector store
    store = VectorStore(dimension=dimension, index_type="flat", mapping_type="json")
    
    # Create some test vectors
    test_vectors = np.random.rand(5, dimension).astype('float32')
    test_metadata = [
        {"text": "First document", "category": "A"},
        {"text": "Second document", "category": "B"},
        {"text": "Third document", "category": "A"},
        {"text": "Fourth document", "category": "C"},
        {"text": "Fifth document", "category": "B"}
    ]
    
    # Add vectors
    vector_ids = store.add_vectors(test_vectors, test_metadata)
    print(f"Added vectors with IDs: {vector_ids}")
    
    # Search for similar vectors
    query = test_vectors[0]  # Use first vector as query
    results = store.search(query, k=3)
    
    print("\nSearch results:")
    for result in results:
        print(f"ID: {result['id']}, Distance: {result['distance']:.4f}, "
              f"Similarity: {result['similarity']:.4f}, Metadata: {result['metadata']}")
    
    # Get stats
    stats = store.get_stats()
    print(f"\nStore stats: {stats}")
