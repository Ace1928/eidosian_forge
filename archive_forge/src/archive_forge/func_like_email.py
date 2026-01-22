import re
import unicodedata
from typing import Set
from .. import attrs
from .tokenizer_exceptions import URL_MATCH
def like_email(text: str) -> bool:
    return bool(_like_email(text))