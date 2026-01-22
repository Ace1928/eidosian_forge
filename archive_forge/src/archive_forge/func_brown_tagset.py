import re
from textwrap import wrap
from nltk.data import load
def brown_tagset(tagpattern=None):
    _format_tagset('brown_tagset', tagpattern)