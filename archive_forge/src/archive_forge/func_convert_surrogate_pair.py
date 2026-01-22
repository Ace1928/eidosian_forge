import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def convert_surrogate_pair(match):
    """
    Convert a surrogate pair to the single codepoint it represents.

    This implements the formula described at:
    http://en.wikipedia.org/wiki/Universal_Character_Set_characters#Surrogates
    """
    pair = match.group(0)
    codept = 65536 + (ord(pair[0]) - 55296) * 1024 + (ord(pair[1]) - 56320)
    return chr(codept)