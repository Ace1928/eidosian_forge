from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
def getNamedResource(self, resType, name):
    """Return the named resource of given type, else return None."""
    name = tostr(name, encoding='mac-roman')
    for res in self.get(resType, []):
        if res.name == name:
            return res
    return None