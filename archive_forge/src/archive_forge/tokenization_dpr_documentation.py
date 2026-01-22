import collections
from typing import List, Optional, Union
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert import BertTokenizer

        Finds the best answer span for the extractive Q&A model for one passage. It returns the best span by descending
        `span_score` order and keeping max `top_spans` spans. Spans longer that `max_answer_length` are ignored.
        