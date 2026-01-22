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
def getNodeIndex(self, node):
    """What is the stream index for node?

    0..n-1
        Return -1 if node not found.
        """
    if self.p == -1:
        self.fillBuffer()
    for i, t in enumerate(self.nodes):
        if t == node:
            return i
    return -1