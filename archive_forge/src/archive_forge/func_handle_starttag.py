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
def handle_starttag(self, tag, attrs):
    """Called by HTMLParser when a tag starts"""
    if isinstance(attrs, list):
        d = {}
        for k, v in attrs:
            d[k] = v
        attrs = d
    if not self.caseSensitive:
        tag = tag.lower()
    try:
        start = getattr(self, 'start_' + tag)
    except AttributeError:
        if not self.ignoreUnknownTags:
            raise ValueError('Invalid tag "%s"' % tag)
        start = self.start_unknown
    start(attrs or {})