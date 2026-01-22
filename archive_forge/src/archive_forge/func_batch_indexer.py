import itertools
import time
from typing import Dict, List, Optional, Union
from redis.client import Pipeline
from redis.utils import deprecated_function
from ..helpers import get_protocol_version, parse_to_dict
from ._util import to_string
from .aggregation import AggregateRequest, AggregateResult, Cursor
from .document import Document
from .query import Query
from .result import Result
from .suggestion import SuggestionParser
def batch_indexer(self, chunk_size=100):
    """
        Create a new batch indexer from the client with a given chunk size
        """
    return self.BatchIndexer(self, chunk_size=chunk_size)