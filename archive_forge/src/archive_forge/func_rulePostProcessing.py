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
def rulePostProcessing(self, root):
    """Transform ^(nil x) to x and nil to null"""
    if root is not None and root.isNil():
        if root.getChildCount() == 0:
            root = None
        elif root.getChildCount() == 1:
            root = root.getChild(0)
            root.setParent(None)
            root.setChildIndex(-1)
    return root