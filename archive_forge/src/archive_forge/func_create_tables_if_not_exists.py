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
def create_tables_if_not_exists(self) -> Any:
    """Create the table to store the texts and embeddings"""
    try:
        from gpudb import GPUdbTable
    except ImportError:
        raise ImportError('Could not import Kinetica python API. Please install it with `pip install gpudb==7.2.0.1`.')
    return GPUdbTable(_type=self.table_schema, name=self.table_name, db=self._db, options={'is_replicated': 'true'})