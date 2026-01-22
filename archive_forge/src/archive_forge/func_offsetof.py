import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def offsetof(self, cdecl, *fields_or_indexes):
    """Return the offset of the named field inside the given
        structure or array, which must be given as a C type name.
        You can give several field names in case of nested structures.
        You can also give numeric values which correspond to array
        items, in case of an array type.
        """
    if isinstance(cdecl, basestring):
        cdecl = self._typeof(cdecl)
    return self._typeoffsetof(cdecl, *fields_or_indexes)[1]