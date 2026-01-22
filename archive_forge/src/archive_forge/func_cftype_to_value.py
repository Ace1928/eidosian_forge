from ctypes import *
from ctypes import util
from .runtime import send_message, ObjCInstance
from .cocoatypes import *
def cftype_to_value(cftype):
    """Convert a CFType into an equivalent python type.
    The convertible CFTypes are taken from the known_cftypes
    dictionary, which may be added to if another library implements
    its own conversion methods."""
    if not cftype:
        return None
    typeID = cf.CFGetTypeID(cftype)
    if typeID in known_cftypes:
        convert_function = known_cftypes[typeID]
        return convert_function(cftype)
    else:
        return cftype