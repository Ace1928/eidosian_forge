import functools
from zaqarclient import errors
def lazy_property(write=False, delete=True):
    """Creates a lazy property.

    :param write: Whether this property is "writable"
    :param delete: Whether this property can be deleted.
    """

    def wrapper(fn):
        attr_name = '_lazy_' + fn.__name__

        def getter(self):
            if not hasattr(self, attr_name):
                setattr(self, attr_name, fn(self))
            return getattr(self, attr_name)

        def setter(self, value):
            setattr(self, attr_name, value)

        def deleter(self):
            delattr(self, attr_name)
        return property(fget=getter, fset=write and setter, fdel=delete and deleter, doc=fn.__doc__)
    return wrapper