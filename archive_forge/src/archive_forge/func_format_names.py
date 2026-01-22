from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_names(self, role, as_sentence=True):
    formatted_names = names(role, sep=', ', sep2=' and ', last_sep=', and ')
    if as_sentence:
        return sentence[formatted_names]
    else:
        return formatted_names