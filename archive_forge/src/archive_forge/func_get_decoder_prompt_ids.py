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
def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
    self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
    forced_tokens = self.prefix_tokens[1:]
    forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
    return forced_decoder_ids