import re
from . import utilities
def remove_optional_outer_square_brackets(text):
    if text[0] == '[':
        return remove_outer_square_brackets(text)
    return text