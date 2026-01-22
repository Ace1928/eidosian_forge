import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def finish_backend_type(self, ffi, finishlist):
    if self.completed:
        if self.completed != 2:
            raise NotImplementedError("recursive structure declaration for '%s'" % (self.name,))
        return
    BType = ffi._cached_btypes[self]
    self.completed = 1
    if self.fldtypes is None:
        pass
    elif self.fixedlayout is None:
        fldtypes = [tp.get_cached_btype(ffi, finishlist) for tp in self.fldtypes]
        lst = list(zip(self.fldnames, fldtypes, self.fldbitsize))
        extra_flags = ()
        if self.packed:
            if self.packed == 1:
                extra_flags = (8,)
            else:
                extra_flags = (0, self.packed)
        ffi._backend.complete_struct_or_union(BType, lst, self, -1, -1, *extra_flags)
    else:
        fldtypes = []
        fieldofs, fieldsize, totalsize, totalalignment = self.fixedlayout
        for i in range(len(self.fldnames)):
            fsize = fieldsize[i]
            ftype = self.fldtypes[i]
            if isinstance(ftype, ArrayType) and ftype.length_is_unknown():
                BItemType = ftype.item.get_cached_btype(ffi, finishlist)
                nlen, nrest = divmod(fsize, ffi.sizeof(BItemType))
                if nrest != 0:
                    self._verification_error("field '%s.%s' has a bogus size?" % (self.name, self.fldnames[i] or '{}'))
                ftype = ftype.resolve_length(nlen)
                self.fldtypes = self.fldtypes[:i] + (ftype,) + self.fldtypes[i + 1:]
            BFieldType = ftype.get_cached_btype(ffi, finishlist)
            if isinstance(ftype, ArrayType) and ftype.length is None:
                assert fsize == 0
            else:
                bitemsize = ffi.sizeof(BFieldType)
                if bitemsize != fsize:
                    self._verification_error("field '%s.%s' is declared as %d bytes, but is really %d bytes" % (self.name, self.fldnames[i] or '{}', bitemsize, fsize))
            fldtypes.append(BFieldType)
        lst = list(zip(self.fldnames, fldtypes, self.fldbitsize, fieldofs))
        ffi._backend.complete_struct_or_union(BType, lst, self, totalsize, totalalignment)
    self.completed = 2