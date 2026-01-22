from __future__ import annotations
import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def process_index_results(self, ids: List[int], scores: List[float], *, k: int=4, filter: Optional[Dict[str, Any]]=None, score_threshold: float=MAX_FLOAT) -> List[Tuple[Document, float]]:
    """Turns TileDB results into a list of documents and scores.

        Args:
            ids: List of indices of the documents in the index.
            scores: List of distances of the documents in the index.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            score_threshold: Optional, a floating point value to filter the
                resulting set of retrieved docs
        Returns:
            List of Documents and scores.
        """
    tiledb_vs, tiledb = dependable_tiledb_import()
    docs = []
    docs_array = tiledb.open(self.docs_array_uri, 'r', timestamp=self.timestamp, config=self.config)
    for idx, score in zip(ids, scores):
        if idx == 0 and score == 0:
            continue
        if idx == MAX_UINT64 and score == MAX_FLOAT_32:
            continue
        doc = docs_array[idx]
        if doc is None or len(doc['text']) == 0:
            raise ValueError(f'Could not find document for id {idx}, got {doc}')
        pickled_metadata = doc.get('metadata')
        result_doc = Document(page_content=str(doc['text'][0]))
        if pickled_metadata is not None:
            metadata = pickle.loads(np.array(pickled_metadata.tolist()).astype(np.uint8).tobytes())
            result_doc.metadata = metadata
        if filter is not None:
            filter = {key: [value] if not isinstance(value, list) else value for key, value in filter.items()}
            if all((result_doc.metadata.get(key) in value for key, value in filter.items())):
                docs.append((result_doc, score))
        else:
            docs.append((result_doc, score))
    docs_array.close()
    docs = [(doc, score) for doc, score in docs if score <= score_threshold]
    return docs[:k]