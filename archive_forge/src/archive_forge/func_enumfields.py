import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def enumfields(self, expand_anonymous_struct_union=True):
    fldquals = self.fldquals
    if fldquals is None:
        fldquals = (0,) * len(self.fldnames)
    for name, type, bitsize, quals in zip(self.fldnames, self.fldtypes, self.fldbitsize, fldquals):
        if name == '' and isinstance(type, StructOrUnion) and expand_anonymous_struct_union:
            for result in type.enumfields():
                yield result
        else:
            yield (name, type, bitsize, quals)