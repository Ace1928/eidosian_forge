import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def handle_quant(self, tok, context):
    factory = self.get_QuantifiedExpression_factory(tok)
    if not self.inRange(0):
        raise ExpectedMoreTokensException(self._currentIndex + 2, message="Variable and Expression expected following quantifier '%s'." % tok)
    vars = [self.get_next_token_variable('quantified')]
    while True:
        if not self.inRange(0) or (self.token(0) == Tokens.DOT and (not self.inRange(1))):
            raise ExpectedMoreTokensException(self._currentIndex + 2, message='Expression expected.')
        if not self.isvariable(self.token(0)):
            break
        vars.append(self.get_next_token_variable('quantified'))
    if self.inRange(0) and self.token(0) == Tokens.DOT:
        self.token()
    accum = self.process_next_expression(tok)
    while vars:
        accum = self.make_QuanifiedExpression(factory, vars.pop(), accum)
    return accum