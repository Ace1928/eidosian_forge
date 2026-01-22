import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def repr_part(self) -> str:
    """FmtStr repr is build by concatenating these."""

    def pp_att(att: str) -> str:
        if att == 'fg':
            return FG_NUMBER_TO_COLOR[self.atts[att]]
        elif att == 'bg':
            return 'on_' + BG_NUMBER_TO_COLOR[self.atts[att]]
        else:
            return att
    atts_out = {k: v for k, v in self._atts.items() if v}
    return ''.join((pp_att(att) + '(' for att in sorted(atts_out))) + repr(self._s) + ')' * len(atts_out)