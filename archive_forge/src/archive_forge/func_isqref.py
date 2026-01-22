from suds import *
from suds.sax import Namespace, splitPrefix
def isqref(object):
    """
    Get whether the object is a I{qualified reference}.
    @param object: An object to be tested.
    @type object: I{any}
    @rtype: boolean
    @see: L{qualify}
    """
    return isinstance(object, tuple) and len(object) == 2 and isinstance(object[0], str) and isinstance(object[1], str)