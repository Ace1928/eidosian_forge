from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_btitle(self, e, which_field, as_sentence=True):
    formatted_title = tag('em')[field(which_field)]
    if as_sentence:
        return sentence[formatted_title]
    else:
        return formatted_title