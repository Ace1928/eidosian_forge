import re
from ..helpers import PREVENT_BACKSLASH
def render_table_head(renderer, text):
    return '<thead>\n<tr>\n' + text + '</tr>\n</thead>\n'