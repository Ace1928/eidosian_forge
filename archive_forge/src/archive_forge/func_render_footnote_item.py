import re
from ..core import BlockState
from ..util import unikey
from ..helpers import LINK_LABEL
def render_footnote_item(renderer, text: str, key: str, index: int):
    i = str(index)
    back = '<a href="#fnref-' + i + '" class="footnote">&#8617;</a>'
    text = text.rstrip()[:-4] + back + '</p>'
    return '<li id="fn-' + i + '">' + text + '</li>\n'