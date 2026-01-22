import re
from ...symbols import ORTH
from ...util import update_exc
from ..char_classes import ALPHA, ALPHA_LOWER
from ..tokenizer_exceptions import BASE_EXCEPTIONS
from .punctuation import ELISION, HYPHENS
def lower_first_letter(text):
    if len(text) == 0:
        return text
    if len(text) == 1:
        return text.lower()
    return text[0].lower() + text[1:]