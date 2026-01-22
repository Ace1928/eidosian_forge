import re
import unicodedata
from typing import Set
from .. import attrs
from .tokenizer_exceptions import URL_MATCH
def is_left_punct(text: str) -> bool:
    left_punct = ('(', '[', '{', '<', '"', "'", '«', '‘', '‚', '‛', '“', '„', '‟', '‹', '❮', '``')
    return text in left_punct