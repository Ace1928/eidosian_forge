import weakref
from weakref import ReferenceType
def resolve_ref(ref):
    return ref() if ref is not None else None