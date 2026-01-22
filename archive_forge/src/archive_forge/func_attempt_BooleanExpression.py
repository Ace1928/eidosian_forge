import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def attempt_BooleanExpression(self, expression, context):
    """Attempt to make a boolean expression.  If the next token is a boolean
        operator, then a BooleanExpression will be returned.  Otherwise, the
        parameter will be returned."""
    while self.inRange(0):
        tok = self.token(0)
        factory = self.get_BooleanExpression_factory(tok)
        if factory and self.has_priority(tok, context):
            self.token()
            expression = self.make_BooleanExpression(factory, expression, self.process_next_expression(tok))
        else:
            break
    return expression