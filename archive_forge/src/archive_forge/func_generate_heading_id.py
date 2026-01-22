from ._base import DirectivePlugin
from ..toc import normalize_toc_item, render_toc_ul
def generate_heading_id(self, token, index):
    return 'toc_' + str(index + 1)