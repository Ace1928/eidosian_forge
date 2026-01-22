import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer
@classmethod
def load_from_disk(cls, vector_size, dataset_path, index_path):
    logger.info(f'Loading passages from {dataset_path}')
    if dataset_path is None or index_path is None:
        raise ValueError("Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` and `dataset.get_index('embeddings').save(index_path)`.")
    dataset = load_from_disk(dataset_path)
    return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)