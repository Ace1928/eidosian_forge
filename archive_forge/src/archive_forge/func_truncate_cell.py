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
def truncate_cell(self, cell_value):
    if isinstance(cell_value, int) or isinstance(cell_value, float):
        return cell_value
    if cell_value.strip() != '':
        try_tokens = self.tokenize(cell_value)
        if len(try_tokens) >= self.max_cell_length:
            retain_tokens = try_tokens[:self.max_cell_length]
            retain_cell_value = self.convert_tokens_to_string(retain_tokens)
            return retain_cell_value
        else:
            return None
    else:
        return cell_value