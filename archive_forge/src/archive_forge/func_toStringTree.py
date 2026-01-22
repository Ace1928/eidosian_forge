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
def toStringTree(self):
    if not self.children:
        return self.toString()
    ret = ''
    if not self.isNil():
        ret += '(%s ' % self.toString()
    ret += ' '.join([child.toStringTree() for child in self.children])
    if not self.isNil():
        ret += ')'
    return ret