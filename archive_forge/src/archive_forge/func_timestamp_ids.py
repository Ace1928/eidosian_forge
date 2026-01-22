import json
import os
import re
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple
import numpy as np
from tokenizers import AddedToken, pre_tokenizers, processors
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from .tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE, WhisperTokenizer, _decode_asr
@lru_cache
def timestamp_ids(self, time_precision=0.02):
    """
        Compute the timestamp token ids for a given precision and save to least-recently used (LRU) cache.

        Args:
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
    return self.convert_tokens_to_ids(['<|%.2f|>' % (i * time_precision) for i in range(1500 + 1)])