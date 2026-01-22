import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def unjellyInto(self, obj, loc, ao):
    """Utility method for unjellying one object into another.
        This automates the handling of backreferences.
        """
    o = self.unjellyAO(ao)
    obj[loc] = o
    if isinstance(o, crefutil.NotKnown):
        o.addDependant(obj, loc)
    return o