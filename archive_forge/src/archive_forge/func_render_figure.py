import re
from ._base import DirectivePlugin
from ..util import escape as escape_text, escape_url
def render_figure(self, text, align=None, figwidth=None, figclass=None):
    _cls = 'figure'
    if align:
        _cls += ' align-' + align
    if figclass:
        _cls += ' ' + figclass
    html = '<figure class="' + _cls + '"'
    if figwidth:
        html += ' style="width:' + figwidth + '"'
    return html + '>\n' + text + '</figure>\n'