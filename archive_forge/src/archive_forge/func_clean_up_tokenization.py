import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .number_normalizer import EnglishNormalizer
def clean_up_tokenization(self, text):
    text = ''.join(text)
    vocab_tokens = list(self.encoder.keys()) + list(self.added_tokens_encoder.keys())
    text = text.replace('[SPACE]', ' ') if '[SPACE]' in vocab_tokens else text
    text = text.replace('[STOP]', ' ') if '[STOP]' in vocab_tokens else text
    text = text.replace(self.unk_token, '').replace('   ', ' ').replace('  ', ' ')
    return text