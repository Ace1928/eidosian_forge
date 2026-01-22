import re
from formencode.rewritingparser import RewritingParser, html_quote
def escape_formatter(error):
    """
    Formatter that escapes HTML, no more.
    """
    return html_quote(error)