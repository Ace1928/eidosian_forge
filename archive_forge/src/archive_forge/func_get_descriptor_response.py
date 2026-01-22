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
def get_descriptor_response(self, command_str: str, setname: str, k_neighbors: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, constraints: Optional[dict]=None, results: Optional[Dict[str, Any]]=None, query_embedding: Optional[List[float]]=None, normalize_distance: bool=False) -> Tuple[List[Dict[str, Any]], List]:
    all_blobs: List[Any] = []
    blob = embedding2bytes(query_embedding)
    if blob is not None:
        all_blobs.append(blob)
    if constraints is None:
        response, response_array, max_dist = self.get_k_candidates(setname, k_neighbors, results, all_blobs, normalize=normalize_distance)
    else:
        if results is None:
            results = {'list': ['id']}
        elif 'list' not in results:
            results['list'] = ['id']
        elif 'id' not in results['list']:
            results['list'].append('id')
        query = _add_descriptor(command_str, setname, constraints=constraints, results=results)
        response, response_array = self.__run_vdms_query([query])
        ids_of_interest = [ent['id'] for ent in response[0][command_str]['entities']]
        response, response_array, max_dist = self.get_k_candidates(setname, fetch_k, results, all_blobs, normalize=normalize_distance)
        new_entities: List[Dict] = []
        for ent in response[0][command_str]['entities']:
            if ent['id'] in ids_of_interest:
                new_entities.append(ent)
            if len(new_entities) == k_neighbors:
                break
        response[0][command_str]['entities'] = new_entities
        response[0][command_str]['returned'] = len(new_entities)
        if len(new_entities) < k_neighbors:
            p_str = 'Returned items < k_neighbors; Try increasing fetch_k'
            print(p_str)
    if normalize_distance:
        max_dist = 1.0 if max_dist == 0 else max_dist
        for ent_idx, ent in enumerate(response[0][command_str]['entities']):
            ent['_distance'] = ent['_distance'] / max_dist
            response[0][command_str]['entities'][ent_idx]['_distance'] = ent['_distance']
    return (response, response_array)