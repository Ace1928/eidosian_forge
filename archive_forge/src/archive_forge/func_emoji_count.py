import re
import unicodedata
from typing import Iterator
from emoji import unicode_codes
from emoji.tokenizer import Token, EmojiMatch, EmojiMatchZWJ, EmojiMatchZWJNonRGI, tokenize, filter_tokens
def emoji_count(string, unique=False):
    """
    Returns the count of emojis in a string.

    :param unique: (optional) True if count only unique emojis
    """
    if unique:
        return len(distinct_emoji_list(string))
    return len(emoji_list(string))