import re
from ._base import DirectivePlugin
from ..util import escape as escape_text, escape_url
def render_legend(self, text):
    return '<div class="legend">\n' + text + '</div>\n'