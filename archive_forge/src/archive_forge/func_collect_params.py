from __future__ import annotations
import enum
import logging
import os
from hashlib import md5
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def collect_params(input_data: List[Tuple[str, Dict[str, str]]]) -> Tuple[List[str], Dict[str, Any]]:
    """Transform the input data into the desired format.

    Args:
    - input_data (list of tuples): Input data to transform.
      Each tuple contains a string and a dictionary.

    Returns:
    - tuple: A tuple containing a list of strings and a dictionary.
    """
    query_parts = []
    params = {}
    for query_part, param in input_data:
        query_parts.append(query_part)
        params.update(param)
    return (query_parts, params)