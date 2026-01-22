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
def delete_unrelated_rows(self, table_content: Dict, question: str, answer: List, delete_ratio: float):
    """
        The argument answer is used only during training.
        """
    truncated_unrelated_indices = []
    related_indices = []
    if answer is None or len(answer) == 0:
        answer_set = set()
    else:
        answer_set = {ans_ex.lower() for ans_ex in answer}
    if question is not None:
        answer_set.update(question.split())
    question_set = set(question.strip('?!.,').split(' '))
    row_max_len = len(table_content['rows'])
    for _row_idx, row in enumerate(table_content['rows']):
        lower_row = {str(cell).lower() for cell in row}
        if len(lower_row & answer_set) == 0 and len(lower_row & question_set) == 0:
            truncated_unrelated_indices.append(_row_idx)
        else:
            related_indices.extend([_row_idx - 2, _row_idx - 1, _row_idx, _row_idx + 1, _row_idx + 2])
    truncated_unrelated_indices = [_row_idx for _row_idx in truncated_unrelated_indices if _row_idx not in related_indices]
    drop_items = min(len(truncated_unrelated_indices), int(len(table_content['rows']) * delete_ratio))
    drop_row_indices = random.choices(truncated_unrelated_indices, k=drop_items)
    for _row_idx in reversed(range(row_max_len)):
        if _row_idx in drop_row_indices:
            del table_content['rows'][_row_idx]
    if 'id' in table_content and len(drop_row_indices) > 0:
        logger.warning('Delete {:.2f} rows in table {}'.format(len(drop_row_indices), table_content['id']))