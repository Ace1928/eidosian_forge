from typing import Callable, Dict, Optional, Sequence
import numpy as np
from langchain_community.document_transformers.embeddings_redundant_filter import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import (
from langchain.utils.math import cosine_similarity
@root_validator()
def validate_params(cls, values: Dict) -> Dict:
    """Validate similarity parameters."""
    if values['k'] is None and values['similarity_threshold'] is None:
        raise ValueError('Must specify one of `k` or `similarity_threshold`.')
    return values