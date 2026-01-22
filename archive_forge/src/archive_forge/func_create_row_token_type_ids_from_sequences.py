import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
def create_row_token_type_ids_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
    """
        Creates the row token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the row token type IDs values.
        """
    table_row_ids = list(zip(*table_values))[2] if table_values else []
    return [0] * (1 + len(query_ids) + 1) + list(table_row_ids)