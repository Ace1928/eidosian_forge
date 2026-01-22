import re
from textwrap import wrap
from nltk.data import load
def claws5_tagset(tagpattern=None):
    _format_tagset('claws5_tagset', tagpattern)