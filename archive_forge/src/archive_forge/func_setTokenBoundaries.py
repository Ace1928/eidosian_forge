from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import UP, DOWN, EOF, INVALID_TOKEN_TYPE
from antlr3.exceptions import MismatchedTreeNodeException, \
from antlr3.recognizers import BaseRecognizer, RuleReturnScope
from antlr3.streams import IntStream
from antlr3.tokens import CommonToken, Token, INVALID_TOKEN
import six
from six.moves import range
def setTokenBoundaries(self, t, startToken, stopToken):
    """
        Track start/stop token for subtree root created for a rule.
        Only works with Tree nodes.  For rules that match nothing,
        seems like this will yield start=i and stop=i-1 in a nil node.
        Might be useful info so I'll not force to be i..i.
        """
    if t is None:
        return
    start = 0
    stop = 0
    if startToken is not None:
        start = startToken.index
    if stopToken is not None:
        stop = stopToken.index
    t.setTokenStartIndex(start)
    t.setTokenStopIndex(stop)