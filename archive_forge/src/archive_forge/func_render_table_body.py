import re
from ..helpers import PREVENT_BACKSLASH
def render_table_body(renderer, text):
    return '<tbody>\n' + text + '</tbody>\n'