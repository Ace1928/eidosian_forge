from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
from antlr3 import runtime_version, runtime_version_str
from antlr3.compat import set, frozenset, reversed
from antlr3.constants import DEFAULT_CHANNEL, HIDDEN_CHANNEL, EOF, \
from antlr3.exceptions import RecognitionException, MismatchedTokenException, \
from antlr3.tokens import CommonToken, EOF_TOKEN, SKIP_TOKEN
import six
from six import unichr
def getTokenErrorDisplay(self, t):
    """
        How should a token be displayed in an error message? The default
        is to display just the text, but during development you might
        want to have a lot of information spit out.  Override in that case
        to use t.toString() (which, for CommonToken, dumps everything about
        the token). This is better than forcing you to override a method in
        your token objects because you don't have to go modify your lexer
        so that it creates a new Java type.
        """
    s = t.text
    if s is None:
        if t.type == EOF:
            s = '<EOF>'
        else:
            s = '<' + t.type + '>'
    return repr(s)