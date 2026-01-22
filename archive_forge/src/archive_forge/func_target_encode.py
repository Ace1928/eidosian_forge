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
def target_encode(self, answer: str, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy, TapexTruncationStrategy]=None, max_length: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> List[int]:
    """
        Prepare the answer string for the model. This method does not return token type IDs, attention masks, etc.
        which are necessary for the model to work correctly. Use this method if you want to build your processing on
        your own, otherwise refer to `__call__`.

        Args:
            answer `str`:
                Corresponding answer supervision to the queries for training the model
        """
    encoded_outputs = self.target_encode_plus(answer=answer, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors, **kwargs)
    return encoded_outputs['input_ids']