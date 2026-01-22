import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def remove_bom(text):
    """
    Remove a byte-order mark that was accidentally decoded as if it were part
    of the text.

    >>> print(remove_bom(chr(0xfeff) + "Where do you want to go today?"))
    Where do you want to go today?
    """
    return text.lstrip(chr(65279))