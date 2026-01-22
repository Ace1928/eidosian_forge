import re
from collections import defaultdict
from operator import itemgetter
from nltk.tree.tree import Tree
from nltk.util import OrderedDict
def dumpmatrix():
    """Dump matrix contents for debugging purposes."""
    return '\n'.join(('%2d: %s' % (n, ' '.join((('%2r' % i)[:2] for i in row))) for n, row in enumerate(matrix)))