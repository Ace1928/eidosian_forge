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
def nextTree(self):
    """
        Return the next element in the stream.  If out of elements, throw
        an exception unless size()==1.  If size is 1, then return elements[0].

        Return a duplicate node/subtree if stream is out of elements and
        size==1. If we've already used the element, dup (dirty bit set).
        """
    if self.dirty or (self.cursor >= len(self) and len(self) == 1):
        el = self._next()
        return self.dup(el)
    el = self._next()
    return el