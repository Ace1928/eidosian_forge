from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
def is_suspiciously_successive_range(unicode_range_a: Optional[str], unicode_range_b: Optional[str]) -> bool:
    """
    Determine if two Unicode range seen next to each other can be considered as suspicious.
    """
    if unicode_range_a is None or unicode_range_b is None:
        return True
    if unicode_range_a == unicode_range_b:
        return False
    if 'Latin' in unicode_range_a and 'Latin' in unicode_range_b:
        return False
    if 'Emoticons' in unicode_range_a or 'Emoticons' in unicode_range_b:
        return False
    if ('Latin' in unicode_range_a or 'Latin' in unicode_range_b) and ('Combining' in unicode_range_a or 'Combining' in unicode_range_b):
        return False
    keywords_range_a, keywords_range_b = (unicode_range_a.split(' '), unicode_range_b.split(' '))
    for el in keywords_range_a:
        if el in UNICODE_SECONDARY_RANGE_KEYWORD:
            continue
        if el in keywords_range_b:
            return False
    range_a_jp_chars, range_b_jp_chars = (unicode_range_a in ('Hiragana', 'Katakana'), unicode_range_b in ('Hiragana', 'Katakana'))
    if (range_a_jp_chars or range_b_jp_chars) and ('CJK' in unicode_range_a or 'CJK' in unicode_range_b):
        return False
    if range_a_jp_chars and range_b_jp_chars:
        return False
    if 'Hangul' in unicode_range_a or 'Hangul' in unicode_range_b:
        if 'CJK' in unicode_range_a or 'CJK' in unicode_range_b:
            return False
        if unicode_range_a == 'Basic Latin' or unicode_range_b == 'Basic Latin':
            return False
    if ('CJK' in unicode_range_a or 'CJK' in unicode_range_b) or (unicode_range_a in ['Katakana', 'Hiragana'] and unicode_range_b in ['Katakana', 'Hiragana']):
        if 'Punctuation' in unicode_range_a or 'Punctuation' in unicode_range_b:
            return False
        if 'Forms' in unicode_range_a or 'Forms' in unicode_range_b:
            return False
    return True