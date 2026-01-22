import types
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydev_bundle._pydev_imports_tipper import signature_from_docstring
def get_bound_class_name(obj):
    my_self = getattr(obj, '__self__', getattr(obj, 'im_self', None))
    if my_self is None:
        return None
    return get_class_name(my_self)