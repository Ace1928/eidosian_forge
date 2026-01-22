import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def process_next_expression(self, context):
    """Parse the next complete expression from the stream and return it."""
    try:
        tok = self.token()
    except ExpectedMoreTokensException as e:
        raise ExpectedMoreTokensException(self._currentIndex + 1, message='Expression expected.') from e
    accum = self.handle(tok, context)
    if not accum:
        raise UnexpectedTokenException(self._currentIndex, tok, message='Expression expected.')
    return self.attempt_adjuncts(accum, context)