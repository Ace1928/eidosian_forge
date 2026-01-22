import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def handle_variable(self, tok, context):
    accum = self.make_VariableExpression(tok)
    if self.inRange(0) and self.token(0) == Tokens.OPEN:
        if not isinstance(accum, FunctionVariableExpression) and (not isinstance(accum, ConstantExpression)):
            raise LogicalExpressionException(self._currentIndex, "'%s' is an illegal predicate name.  Individual variables may not be used as predicates." % tok)
        self.token()
        accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
        while self.inRange(0) and self.token(0) == Tokens.COMMA:
            self.token()
            accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
        self.assertNextToken(Tokens.CLOSE)
    return accum