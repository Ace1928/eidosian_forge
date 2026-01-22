import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def original_text_for(expr: ParserElement, as_string: bool=True, *, asString: bool=True) -> ParserElement:
    """Helper to return the original, untokenized text for a given
    expression.  Useful to restore the parsed fields of an HTML start
    tag into the raw tag text itself, or to revert separate tokens with
    intervening whitespace back to the original matching input text. By
    default, returns a string containing the original parsed text.

    If the optional ``as_string`` argument is passed as
    ``False``, then the return value is
    a :class:`ParseResults` containing any results names that
    were originally matched, and a single token containing the original
    matched text from the input string.  So if the expression passed to
    :class:`original_text_for` contains expressions with defined
    results names, you must set ``as_string`` to ``False`` if you
    want to preserve those results name values.

    The ``asString`` pre-PEP8 argument is retained for compatibility,
    but will be removed in a future release.

    Example::

        src = "this is test <b> bold <i>text</i> </b> normal text "
        for tag in ("b", "i"):
            opener, closer = make_html_tags(tag)
            patt = original_text_for(opener + ... + closer)
            print(patt.search_string(src)[0])

    prints::

        ['<b> bold <i>text</i> </b>']
        ['<i>text</i>']
    """
    asString = asString and as_string
    locMarker = Empty().set_parse_action(lambda s, loc, t: loc)
    endlocMarker = locMarker.copy()
    endlocMarker.callPreparse = False
    matchExpr = locMarker('_original_start') + expr + endlocMarker('_original_end')
    if asString:
        extractText = lambda s, l, t: s[t._original_start:t._original_end]
    else:

        def extractText(s, l, t):
            t[:] = [s[t.pop('_original_start'):t.pop('_original_end')]]
    matchExpr.set_parse_action(extractText)
    matchExpr.ignoreExprs = expr.ignoreExprs
    matchExpr.suppress_warning(Diagnostics.warn_ungrouped_named_tokens_in_collection)
    return matchExpr