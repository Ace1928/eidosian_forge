import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union
from tokenizers import processors
from ...tokenization_utils import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, is_sentencepiece_available, logging
@tgt_lang.setter
def tgt_lang(self, new_tgt_lang: str) -> None:
    if '__' not in new_tgt_lang:
        self._tgt_lang = f'__{new_tgt_lang}__'
    else:
        self._tgt_lang = new_tgt_lang
    self.set_tgt_lang_special_tokens(self._tgt_lang)