import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def fix_embedded_mojibake(match):
    substr = match.group(0)
    if len(substr) < len(text) and is_bad(substr):
        return ftfy.fix_encoding(substr)
    else:
        return substr