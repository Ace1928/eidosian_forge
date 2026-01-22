from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
def getIndResource(self, resType, index):
    """Return resource of given type located at an index ranging from 1
        to the number of resources for that type, or None if not found.
        """
    if index < 1:
        return None
    try:
        res = self[resType][index - 1]
    except (KeyError, IndexError):
        return None
    return res