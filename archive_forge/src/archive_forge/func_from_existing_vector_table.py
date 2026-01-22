import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
@classmethod
def from_existing_vector_table(cls, embedding: Embeddings, connection_string: str, table_name: str, distance_strategy: str=DEFAULT_DISTANCE_STRATEGY, *, engine_args: Optional[Dict[str, Any]]=None, **kwargs: Any) -> VectorStore:
    """
        Create a VectorStore instance from an existing TiDB Vector Store in TiDB.

        Args:
            embedding (Embeddings): The function to use for generating embeddings.
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            table_name (str, optional): The name of table used to store vector data,
                defaults to "langchain_vector".
            distance_strategy: The distance strategy used for similarity search,
                defaults to "cosine", allowed: "l2", "cosine", 'inner_product'.
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            **kwargs (Any): Additional keyword arguments.
        Returns:
            VectorStore: The VectorStore instance.

        Raises:
            NoSuchTableError: If the specified table does not exist in the TiDB.
        """
    try:
        from tidb_vector.integrations import check_table_existence
    except ImportError:
        raise ImportError('Could not import tidbvec python package. Please install it with `pip install tidb-vector`.')
    if check_table_existence(connection_string, table_name):
        return cls(connection_string=connection_string, table_name=table_name, embedding_function=embedding, distance_strategy=distance_strategy, engine_args=engine_args, **kwargs)
    else:
        raise ValueError(f'Table {table_name} does not exist in the TiDB database.')