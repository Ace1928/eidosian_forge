import ctypes
import weakref
from OpenGL._bytes import long, integer_types
def originalObject(self, voidPointer):
    """Given a void-pointer, try to find our original Python object"""
    if isinstance(voidPointer, integer_types):
        identity = voidPointer
    elif voidPointer is None:
        return None
    else:
        try:
            identity = voidPointer.value
        except AttributeError as err:
            identity = voidPointer[0]
    try:
        return self.dataPointers[identity]
    except (KeyError, AttributeError) as err:
        return voidPointer