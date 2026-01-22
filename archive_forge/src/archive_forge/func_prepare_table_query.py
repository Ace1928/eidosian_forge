import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging
def prepare_table_query(self, table, query, answer=None, truncation_strategy=Union[str, TruncationStrategy, TapexTruncationStrategy], max_length=None):
    """
        This method can be used to linearize a table and add a corresponding query.

        Optionally, it also handles truncation of the table (cells).

        An answer can be provided for more precise truncation.
        """
    if not table.empty:
        table_content = {'header': list(table.columns), 'rows': [list(row.values) for i, row in table.iterrows()]}
        self.truncate_table_cells(table_content, query, answer)
        if truncation_strategy == TapexTruncationStrategy.DROP_ROWS_TO_FIT:
            self.truncate_table_rows(table_content, query, answer, max_length=max_length)
        linear_table = self.table_linearize.process_table(table_content)
    else:
        linear_table = ''
    if linear_table == '':
        logger.warning('You provide an empty table, or all cells contain much tokens (e.g., >= 1024 tokens). ' + f'Please carefully check the corresponding table with the query : {query}.')
    if query == '':
        logger.warning('You provide nothing to query with respect to the table.')
    separator = ' ' if query and linear_table else ''
    joint_input = query + separator + linear_table if query else linear_table
    return joint_input