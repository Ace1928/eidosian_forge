import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def is_wildcard(self, p):
    if isinstance(p, (list, tuple)) and len(p) == 1:
        p, = p
    if isinstance(p, gast.Name) and p.id == '_':
        return True
    if p == '_':
        return True
    return False