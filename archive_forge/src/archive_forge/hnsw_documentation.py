from __future__ import annotations
from typing import Any, List, Literal, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.docarray.base import (
Create an DocArrayHnswSearch store and insert data.


        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            **kwargs: Other keyword arguments to be passed to the __init__ method.

        Returns:
            DocArrayHnswSearch Vector Store
        