import os
import re
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_torch_available, logging
def decode_fast(self, token_ids: Union[int, List[int]]) -> str:
    """
        Encodes a text or batch of texts to token ids using preprocessing and the raw SP tokenizer. This has reduced
        functionality but is often much faster.

        Args:
            token_ids (`int` or `List[int]`): Encoded token or text as token id(s).

        Returns:
            `str`: Decoded text
        """
    return self.sp_model.decode(token_ids)