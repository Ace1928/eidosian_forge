import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def handle_open(self, tok, context):
    accum = self.process_next_expression(None)
    self.assertNextToken(Tokens.CLOSE)
    return accum