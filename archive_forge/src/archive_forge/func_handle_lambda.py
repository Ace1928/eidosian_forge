import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def handle_lambda(self, tok, context):
    if not self.inRange(0):
        raise ExpectedMoreTokensException(self._currentIndex + 2, message='Variable and Expression expected following lambda operator.')
    vars = [self.get_next_token_variable('abstracted')]
    while True:
        if not self.inRange(0) or (self.token(0) == Tokens.DOT and (not self.inRange(1))):
            raise ExpectedMoreTokensException(self._currentIndex + 2, message='Expression expected.')
        if not self.isvariable(self.token(0)):
            break
        vars.append(self.get_next_token_variable('abstracted'))
    if self.inRange(0) and self.token(0) == Tokens.DOT:
        self.token()
    accum = self.process_next_expression(tok)
    while vars:
        accum = self.make_LambdaExpression(vars.pop(), accum)
    return accum