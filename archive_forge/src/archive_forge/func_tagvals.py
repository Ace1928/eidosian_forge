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
def tagvals(self, tagfield: str):
    """
        Return a list of all possible tag values

        ### Parameters

        - **tagfield**: Tag field name

        For more information see `FT.TAGVALS <https://redis.io/commands/ft.tagvals>`_.
        """
    return self.execute_command(TAGVALS_CMD, self.index_name, tagfield)