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
def truncate_table_cells(self, table_content: Dict, question: str, answer: List):
    cell_mapping = {}
    for row in table_content['rows']:
        for i, cell in enumerate(row):
            truncate_cell = self.truncate_cell(cell)
            if truncate_cell is not None:
                cell_mapping[cell] = truncate_cell
                row[i] = truncate_cell
    if answer is not None:
        for i, case in enumerate(answer):
            if case in cell_mapping.keys():
                answer[i] = cell_mapping[case]