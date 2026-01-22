import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def markup_serialize_tokens(tokens, markup_func):
    """
    Serialize the list of tokens into a list of text chunks, calling
    markup_func around text to add annotations.
    """
    for token in tokens:
        yield from token.pre_tags
        html = token.html()
        html = markup_func(html, token.annotation)
        if token.trailing_whitespace:
            html += token.trailing_whitespace
        yield html
        yield from token.post_tags