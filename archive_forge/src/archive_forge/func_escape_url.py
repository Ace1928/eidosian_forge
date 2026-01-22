import re
from urllib.parse import quote
from html import _replace_charref
def escape_url(link: str):
    """Escape URL for safety."""
    safe = ':/?#@!$&()*+,;=%'
    return escape(quote(unescape(link), safe=safe))