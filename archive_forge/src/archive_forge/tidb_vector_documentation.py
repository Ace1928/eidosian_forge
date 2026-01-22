import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

        Select the relevance score function based on the distance strategy.
        