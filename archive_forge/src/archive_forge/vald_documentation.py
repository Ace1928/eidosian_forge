from __future__ import annotations
from typing import Any, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance

        Args:
            skip_strict_exist_check: Deprecated. This is not used basically.
        