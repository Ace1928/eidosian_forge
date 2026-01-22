from __future__ import annotations
import json
import logging
import uuid
import warnings
from itertools import repeat
from typing import (
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
Perform a similarity search with Yellowbrick by vectors

        Args:
            embedding (List[float]): query embedding
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document]: List of documents
        