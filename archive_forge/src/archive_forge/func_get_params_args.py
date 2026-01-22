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
def get_params_args(self, query_params: Union[Dict[str, Union[str, int, float, bytes]], None]):
    if query_params is None:
        return []
    args = []
    if len(query_params) > 0:
        args.append('params')
        args.append(len(query_params) * 2)
        for key, value in query_params.items():
            args.append(key)
            args.append(value)
    return args