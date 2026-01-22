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
def dropindex(self, delete_documents: bool=False):
    """
        Drop the index if it exists.
        Replaced `drop_index` in RediSearch 2.0.
        Default behavior was changed to not delete the indexed documents.

        ### Parameters:

        - **delete_documents**: If `True`, all documents will be deleted.

        For more information see `FT.DROPINDEX <https://redis.io/commands/ft.dropindex>`_.
        """
    delete_str = 'DD' if delete_documents else ''
    return self.execute_command(DROPINDEX_CMD, self.index_name, delete_str)