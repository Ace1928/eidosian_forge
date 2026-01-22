import collections
import itertools
import json
import os
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import add_end_docstrings, logging
def convert_tokens_to_shape_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
    if tokens is None:
        return None
    ids = []
    for token in tokens:
        ids.append(self._convert_token_to_shape_id(token))
    return ids