from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@property
def distance_strategy(self) -> str:
    if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
        return 'l2_distance'
    elif self._distance_strategy == DistanceStrategy.COSINE:
        return 'cosine_distance'
    elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
        return 'dot_product'
    else:
        raise ValueError(f'Got unexpected value for distance: {self._distance_strategy}. Should be one of {', '.join([ds.value for ds in DistanceStrategy])}.')