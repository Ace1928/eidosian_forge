import re
from formencode.rewritingparser import RewritingParser, html_quote
def none_formatter(error):
    """
    Formatter that does nothing, no escaping HTML, nothin'
    """
    return error