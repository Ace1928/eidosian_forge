import ctypes, ctypes.util, operator, sys
from . import model
def typeoffsetof(self, BType, fieldname, num=0):
    if isinstance(fieldname, str):
        if num == 0 and issubclass(BType, CTypesGenericPtr):
            BType = BType._BItem
        if not issubclass(BType, CTypesBaseStructOrUnion):
            raise TypeError('expected a struct or union ctype')
        BField = BType._bfield_types[fieldname]
        if BField is Ellipsis:
            raise TypeError('not supported for bitfields')
        return (BField, BType._offsetof(fieldname))
    elif isinstance(fieldname, (int, long)):
        if issubclass(BType, CTypesGenericArray):
            BType = BType._CTPtr
        if not issubclass(BType, CTypesGenericPtr):
            raise TypeError('expected an array or ptr ctype')
        BItem = BType._BItem
        offset = BItem._get_size() * fieldname
        if offset > sys.maxsize:
            raise OverflowError
        return (BItem, offset)
    else:
        raise TypeError(type(fieldname))