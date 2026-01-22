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
def dupTree(self, t, parent=None):
    """
        This is generic in the sense that it will work with any kind of
        tree (not just Tree interface).  It invokes the adaptor routines
        not the tree node routines to do the construction.
        """
    if t is None:
        return None
    newTree = self.dupNode(t)
    self.setChildIndex(newTree, self.getChildIndex(t))
    self.setParent(newTree, parent)
    for i in range(self.getChildCount(t)):
        child = self.getChild(t, i)
        newSubTree = self.dupTree(child, t)
        self.addChild(newTree, newSubTree)
    return newTree