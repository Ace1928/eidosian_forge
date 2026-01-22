from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def get_k_candidates(self, setname: str, fetch_k: Optional[int], results: Optional[Dict[str, Any]]=None, all_blobs: Optional[List]=None, normalize: Optional[bool]=False) -> Tuple[List[Dict[str, Any]], List, float]:
    max_dist = 1
    command_str = 'FindDescriptor'
    query = _add_descriptor(command_str, setname, k_neighbors=fetch_k, results=results)
    response, response_array = self.__run_vdms_query([query], all_blobs)
    if normalize:
        max_dist = response[0][command_str]['entities'][-1]['_distance']
    return (response, response_array, max_dist)