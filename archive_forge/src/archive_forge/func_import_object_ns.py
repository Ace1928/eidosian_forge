import sys
import traceback
def import_object_ns(name_space, import_str, *args, **kwargs):
    """Tries to import object from default namespace.

    Imports a class and return an instance of it, first by trying
    to find the class in a default namespace, then failing back to
    a full path if not found in the default namespace.

    .. versionadded:: 0.3

    .. versionchanged:: 2.6
       Don't capture :exc:`ImportError` when instanciating the object, only
       when importing the object class.
    """
    import_value = '%s.%s' % (name_space, import_str)
    try:
        cls = import_class(import_value)
    except ImportError:
        cls = import_class(import_str)
    return cls(*args, **kwargs)