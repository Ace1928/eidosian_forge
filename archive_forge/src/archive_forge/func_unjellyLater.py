import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def unjellyLater(self, node):
    """Unjelly a node, later."""
    d = crefutil._Defer()
    self.unjellyInto(d, 0, node)
    return d