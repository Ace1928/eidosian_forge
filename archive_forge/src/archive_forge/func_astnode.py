import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes
def astnode(self, s):
    """Return a Python3 ast Node compiled from a string."""
    try:
        import ast
    except ImportError:
        return eval(s)
    p = ast.parse('__tempvalue__ = ' + s)
    return p.body[0].value