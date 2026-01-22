import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def moses_detokenize(self, tokens, lang):
    if lang not in self.cache_moses_detokenizer:
        moses_detokenizer = self.sm.MosesDetokenizer(lang=lang)
        self.cache_moses_detokenizer[lang] = moses_detokenizer
    return self.cache_moses_detokenizer[lang].detokenize(tokens)