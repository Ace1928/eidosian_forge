import re
import string
from pathlib import Path
from typing import Any, Dict, Union
import srsly
from ... import util
from ...language import BaseDefaults, Language
from ...tokens import Doc
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
def pyvi_sylabelize_with_ws(self, text):
    """Modified from pyvi to preserve whitespace and skip unicode
        normalization."""
    specials = ['==>', '->', '\\.\\.\\.', '>>']
    digit = '\\d+([\\.,_]\\d+)+'
    email = '([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\\.)+[a-zA-Z0-9-]+)'
    web = '\\w+://[^\\s]+'
    word = '\\w+'
    non_word = '[^\\w\\s]'
    abbreviations = ['[A-ZĐ]+\\.', 'Tp\\.', 'Mr\\.', 'Mrs\\.', 'Ms\\.', 'Dr\\.', 'ThS\\.']
    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend([digit, non_word, word])
    patterns = '(\\s+|' + '|'.join(patterns) + ')'
    tokens = re.findall(patterns, text, re.UNICODE)
    return [token[0] for token in tokens]