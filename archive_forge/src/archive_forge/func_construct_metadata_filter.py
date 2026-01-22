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
def construct_metadata_filter(filter: Dict[str, Any]) -> Tuple[str, Dict]:
    """Construct a metadata filter.

    Args:
        filter: A dictionary representing the filter condition.

    Returns:
        Tuple[str, Dict]
    """
    if isinstance(filter, dict):
        if len(filter) == 1:
            key, value = list(filter.items())[0]
            if key.startswith('$'):
                if key.lower() not in ['$and', '$or']:
                    raise ValueError(f'Invalid filter condition. Expected $and or $or but got: {key}')
            else:
                return _handle_field_filter(key, filter[key])
            if not isinstance(value, list):
                raise ValueError(f'Expected a list, but got {type(value)} for value: {value}')
            if key.lower() == '$and':
                and_ = combine_queries([construct_metadata_filter(el) for el in value], 'AND')
                if len(and_) >= 1:
                    return and_
                else:
                    raise ValueError('Invalid filter condition. Expected a dictionary but got an empty dictionary')
            elif key.lower() == '$or':
                or_ = combine_queries([construct_metadata_filter(el) for el in value], 'OR')
                if len(or_) >= 1:
                    return or_
                else:
                    raise ValueError('Invalid filter condition. Expected a dictionary but got an empty dictionary')
            else:
                raise ValueError(f'Invalid filter condition. Expected $and or $or but got: {key}')
        elif len(filter) > 1:
            for key in filter.keys():
                if key.startswith('$'):
                    raise ValueError(f'Invalid filter condition. Expected a field but got: {key}')
            and_multiple = collect_params([_handle_field_filter(k, v, index) for index, (k, v) in enumerate(filter.items())])
            if len(and_multiple) >= 1:
                return (' AND '.join(and_multiple[0]), and_multiple[1])
            else:
                raise ValueError('Invalid filter condition. Expected a dictionary but got an empty dictionary')
        else:
            raise ValueError('Got an empty dictionary for filters.')