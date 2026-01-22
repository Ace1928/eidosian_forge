import logging
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
@staticmethod
def get_dense_specific_model_similarity_params(search_params: Dict[str, Any]) -> Dict:
    model = search_params.get('model', 'exact')
    similarity = search_params.get('similarity', 'cosine')
    specific_params = {'model': model, 'similarity': similarity}
    if not model == 'exact':
        if model not in ('lsh', 'permutation_lsh'):
            raise ValueError(f"vector type knn_dense_float_vector doesn't support model {model}")
        if similarity not in ('cosine', 'l2'):
            raise ValueError(f"model exact doesn't support similarity {similarity}")
        specific_params['candidates'] = search_params.get('candidates', search_params.get('size', 4))
        if model == 'lsh' and similarity == 'l2':
            specific_params['probes'] = search_params.get('probes', 0)
    elif similarity not in ('cosine', 'l2'):
        raise ValueError(f"model exact don't support similarity {similarity}")
    return specific_params