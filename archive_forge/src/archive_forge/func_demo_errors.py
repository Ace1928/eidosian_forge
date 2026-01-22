import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def demo_errors():
    print('=' * 20 + 'Test reader errors' + '=' * 20)
    demoException('(P(x) & Q(x)')
    demoException('((P(x) &) & Q(x))')
    demoException('P(x) -> ')
    demoException('P(x')
    demoException('P(x,')
    demoException('P(x,)')
    demoException('exists')
    demoException('exists x.')
    demoException('\\')
    demoException('\\ x y.')
    demoException('P(x)Q(x)')
    demoException('(P(x)Q(x)')
    demoException('exists x -> y')