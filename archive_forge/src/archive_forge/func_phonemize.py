import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (
def phonemize(self, text: str, phonemizer_lang: Optional[str]=None) -> str:
    from phonemizer.separator import Separator
    word_delimiter = self.word_delimiter_token + ' ' if self.word_delimiter_token is not None else ''
    if phonemizer_lang is not None and phonemizer_lang != self.phonemizer_lang:
        self.init_backend(phonemizer_lang)
    else:
        phonemizer_lang = self.phonemizer_lang
    separator = Separator(phone=self.phone_delimiter_token, word=word_delimiter, syllable='')
    phonemes = self.backend.phonemize([text], separator=separator)
    phonemes = phonemes[0].strip()
    return phonemes