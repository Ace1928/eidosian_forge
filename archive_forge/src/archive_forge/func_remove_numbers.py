import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def remove_numbers(lines):

    def _clean(s):
        return re.sub('(?:[\\d_]|\\*\\*)', '', s).strip()
    if isinstance(lines, str):
        return _clean(lines)
    out = []
    for l in lines:
        out.append(_clean(l))
    return out