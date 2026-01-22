import types
from jedi import debug
Retrieve attributes without triggering dynamic lookup via the
       descriptor protocol,  __getattr__ or __getattribute__.

       Note: this function may not be able to retrieve all attributes
       that getattr can fetch (like dynamically created attributes)
       and may find attributes that getattr can't (like descriptors
       that raise AttributeError). It can also return descriptor objects
       instead of instance members in some cases. See the
       documentation for details.

       Returns a tuple `(attr, is_get_descriptor)`. is_get_descripter means that
       the attribute is a descriptor that has a `__get__` attribute.
    