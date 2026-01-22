import re
import types
from ..util import escape
from ..helpers import PREVENT_BACKSLASH
def render_abbr(renderer, text, title):
    if not title:
        return '<abbr>' + text + '</abbr>'
    return '<abbr title="' + escape(title) + '">' + text + '</abbr>'