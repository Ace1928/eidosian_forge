import re
from textwrap import wrap
from nltk.data import load
def upenn_tagset(tagpattern=None):
    _format_tagset('upenn_tagset', tagpattern)