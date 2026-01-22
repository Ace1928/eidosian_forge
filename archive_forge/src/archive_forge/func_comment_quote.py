import html
import html.entities
import re
from urllib.parse import quote, unquote
def comment_quote(s):
    """
    Quote that makes sure text can't escape a comment
    """
    comment = str(s)
    comment = _comment_quote_re.sub('-&gt;', comment)
    return comment