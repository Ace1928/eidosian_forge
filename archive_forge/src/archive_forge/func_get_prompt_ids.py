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
def get_prompt_ids(self, text: str, return_tensors='np'):
    """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
    batch_encoding = self('<|startofprev|>', ' ' + text.strip(), add_special_tokens=False)
    prompt_text_ids = batch_encoding['input_ids'][1:]
    special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)
    if special_token_id is not None:
        token = self.convert_ids_to_tokens(special_token_id)
        raise ValueError(f'Encountered text in the prompt corresponding to disallowed special token: {token}.')
    batch_encoding.convert_to_tensors(tensor_type=return_tensors)
    return batch_encoding['input_ids']