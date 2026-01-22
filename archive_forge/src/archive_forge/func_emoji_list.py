import re
import unicodedata
from typing import Iterator
from emoji import unicode_codes
from emoji.tokenizer import Token, EmojiMatch, EmojiMatchZWJ, EmojiMatchZWJNonRGI, tokenize, filter_tokens
def emoji_list(string):
    """
    Returns the location and emoji in list of dict format.
        >>> emoji.emoji_list("Hi, I am fine. 😁")
        [{'match_start': 15, 'match_end': 16, 'emoji': '😁'}]
    """
    return [{'match_start': m.value.start, 'match_end': m.value.end, 'emoji': m.value.emoji} for m in tokenize(string, keep_zwj=False) if isinstance(m.value, EmojiMatch)]