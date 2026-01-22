import ctypes
def opaque_pointer_cls(name):
    """Create an Opaque pointer class for the given name"""
    typ = type(name, (_Opaque,), {})
    p_typ = type(name + '_pointer', (_opaque_pointer,), {'_type_': typ})
    return p_typ