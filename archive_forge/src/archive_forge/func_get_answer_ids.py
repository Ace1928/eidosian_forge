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
def get_answer_ids(self, column_ids, row_ids, tokenized_table, answer_texts_question, answer_coordinates_question):
    if self.update_answer_coordinates:
        return self._find_answer_ids_from_answer_texts(column_ids, row_ids, tokenized_table, answer_texts=[self.tokenize(at) for at in answer_texts_question])
    return self._get_answer_ids(column_ids, row_ids, answer_coordinates_question)