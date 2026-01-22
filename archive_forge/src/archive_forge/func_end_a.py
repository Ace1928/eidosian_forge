import re
import sys
import copy
import unicodedata
import reportlab.lib.sequencer
from reportlab.lib.abag import ABag
from reportlab.lib.utils import ImageReader, annotateException, encode_label, asUnicode
from reportlab.lib.colors import toColor, black
from reportlab.lib.fonts import tt2ps, ps2tt
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch,mm,cm,pica
from reportlab.rl_config import platypus_link_underline
from html.parser import HTMLParser
from html.entities import name2codepoint
def end_a(self):
    frag = self._stack[-1]
    sct = getattr(frag, '_selfClosingTag', '')
    if sct:
        if not (sct == 'anchor' and frag.name):
            raise ValueError('Parser failure in <a/>')
        defn = frag.cbDefn = ABag()
        defn.label = defn.kind = 'anchor'
        defn.name = frag.name
        del frag.name, frag._selfClosingTag
        self.handle_data('')
        self._pop('a')
    elif self._pop('a').link is None:
        raise ValueError('<link> has no href')