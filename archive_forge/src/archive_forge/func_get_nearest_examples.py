import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def get_nearest_examples(self, index_name: str, query: Union[str, np.array], k: int=10, **kwargs) -> NearestExamplesResults:
    """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The index_name/identifier of the index.
            query (`Union[str, np.ndarray]`):
                The query as a string if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve.

        Returns:
            `(scores, examples)`:
                A tuple of `(scores, examples)` where:
                - **scores** (`List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples
                - **examples** (`dict`): the retrieved examples
        """
    self._check_index_is_initialized(index_name)
    scores, indices = self.search(index_name, query, k, **kwargs)
    top_indices = [i for i in indices if i >= 0]
    return NearestExamplesResults(scores[:len(top_indices)], self[top_indices])