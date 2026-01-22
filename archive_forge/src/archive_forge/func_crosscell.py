import re
from collections import defaultdict
from operator import itemgetter
from nltk.tree.tree import Tree
from nltk.util import OrderedDict
def crosscell(cur, x=vertline):
    """Overwrite center of this cell with a vertical branch."""
    splitl = len(cur) - len(cur) // 2 - len(x) // 2 - 1
    lst = list(cur)
    lst[splitl:splitl + len(x)] = list(x)
    return ''.join(lst)