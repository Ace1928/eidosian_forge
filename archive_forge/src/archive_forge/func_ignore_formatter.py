import re
from formencode.rewritingparser import RewritingParser, html_quote
def ignore_formatter(error):
    """
    Formatter that emits nothing, regardless of the error.
    """
    return ''