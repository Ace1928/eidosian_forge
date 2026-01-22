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
def errorNode(self, input, start, stop, exc):
    """
        create tree node that holds the start and stop tokens associated
        with an error.

        If you specify your own kind of tree nodes, you will likely have to
        override this method. CommonTree returns Token.INVALID_TOKEN_TYPE
        if no token payload but you might have to set token type for diff
        node type.
        """
    return CommonErrorNode(input, start, stop, exc)