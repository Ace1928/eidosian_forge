import re
from ..util import strip_end
def render_def_list_item(renderer, text):
    return '<dd>' + text + '</dd>\n'