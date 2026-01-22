import re
from ..helpers import PREVENT_BACKSLASH
def render_insert(renderer, text):
    return '<ins>' + text + '</ins>'