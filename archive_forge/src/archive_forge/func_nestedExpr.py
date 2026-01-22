import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def nestedExpr(opener='(', closer=')', content=None, ignoreExpr=quotedString.copy()):
    """Helper method for defining nested lists enclosed in opening and closing
       delimiters ("(" and ")" are the default).

       Parameters:
        - opener - opening character for a nested list (default="("); can also be a pyparsing expression
        - closer - closing character for a nested list (default=")"); can also be a pyparsing expression
        - content - expression for items within the nested lists (default=None)
        - ignoreExpr - expression for ignoring opening and closing delimiters (default=quotedString)

       If an expression is not provided for the content argument, the nested
       expression will capture all whitespace-delimited content between delimiters
       as a list of separate values.

       Use the C{ignoreExpr} argument to define expressions that may contain
       opening or closing characters that should not be treated as opening
       or closing characters for nesting, such as quotedString or a comment
       expression.  Specify multiple expressions using an C{L{Or}} or C{L{MatchFirst}}.
       The default is L{quotedString}, but if no expressions are to be ignored,
       then pass C{None} for this argument.
    """
    if opener == closer:
        raise ValueError('opening and closing strings cannot be the same')
    if content is None:
        if isinstance(opener, basestring) and isinstance(closer, basestring):
            if len(opener) == 1 and len(closer) == 1:
                if ignoreExpr is not None:
                    content = Combine(OneOrMore(~ignoreExpr + CharsNotIn(opener + closer + ParserElement.DEFAULT_WHITE_CHARS, exact=1))).setParseAction(lambda t: t[0].strip())
                else:
                    content = empty.copy() + CharsNotIn(opener + closer + ParserElement.DEFAULT_WHITE_CHARS).setParseAction(lambda t: t[0].strip())
            elif ignoreExpr is not None:
                content = Combine(OneOrMore(~ignoreExpr + ~Literal(opener) + ~Literal(closer) + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1))).setParseAction(lambda t: t[0].strip())
            else:
                content = Combine(OneOrMore(~Literal(opener) + ~Literal(closer) + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1))).setParseAction(lambda t: t[0].strip())
        else:
            raise ValueError('opening and closing arguments must be strings if no content expression is given')
    ret = Forward()
    if ignoreExpr is not None:
        ret << Group(Suppress(opener) + ZeroOrMore(ignoreExpr | ret | content) + Suppress(closer))
    else:
        ret << Group(Suppress(opener) + ZeroOrMore(ret | content) + Suppress(closer))
    return ret