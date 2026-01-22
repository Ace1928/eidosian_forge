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
def dict_del(self, name: str, *terms: List[str]):
    """Deletes terms from a dictionary.

        ### Parameters

        - **name**: Dictionary name.
        - **terms**: List of items for removing from the dictionary.

        For more information see `FT.DICTDEL <https://redis.io/commands/ft.dictdel>`_.
        """
    cmd = [DICT_DEL_CMD, name]
    cmd.extend(terms)
    return self.execute_command(*cmd)