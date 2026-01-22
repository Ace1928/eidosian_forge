import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def parse_html(html, cleanup=True):
    """
    Parses an HTML fragment, returning an lxml element.  Note that the HTML will be
    wrapped in a <div> tag that was not in the original document.

    If cleanup is true, make sure there's no <head> or <body>, and get
    rid of any <ins> and <del> tags.
    """
    if cleanup:
        html = cleanup_html(html)
    return fragment_fromstring(html, create_parent=True)