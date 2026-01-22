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
def sugdel(self, key: str, string: str) -> int:
    """
        Delete a string from the AutoCompleter index.
        Returns 1 if the string was found and deleted, 0 otherwise.

        For more information see `FT.SUGDEL <https://redis.io/commands/ft.sugdel>`_.
        """
    return self.execute_command(SUGDEL_COMMAND, key, string)