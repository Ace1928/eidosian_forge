import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def pp_att(att: str) -> str:
    if att == 'fg':
        return FG_NUMBER_TO_COLOR[self.atts[att]]
    elif att == 'bg':
        return 'on_' + BG_NUMBER_TO_COLOR[self.atts[att]]
    else:
        return att