from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
Construct USearch wrapper from raw documents.
        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the USearch database
        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import USearch
                from langchain_community.embeddings import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                usearch = USearch.from_texts(texts, embeddings)
        