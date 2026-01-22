import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def make_html_tags(tag_str: Union[str, ParserElement]) -> Tuple[ParserElement, ParserElement]:
    """Helper to construct opening and closing tag expressions for HTML,
    given a tag name. Matches tags in either upper or lower case,
    attributes with namespaces and with quoted or unquoted values.

    Example::

        text = '<td>More info at the <a href="https://github.com/pyparsing/pyparsing/wiki">pyparsing</a> wiki page</td>'
        # make_html_tags returns pyparsing expressions for the opening and
        # closing tags as a 2-tuple
        a, a_end = make_html_tags("A")
        link_expr = a + SkipTo(a_end)("link_text") + a_end

        for link in link_expr.search_string(text):
            # attributes in the <A> tag (like "href" shown here) are
            # also accessible as named results
            print(link.link_text, '->', link.href)

    prints::

        pyparsing -> https://github.com/pyparsing/pyparsing/wiki
    """
    return _makeTags(tag_str, False)