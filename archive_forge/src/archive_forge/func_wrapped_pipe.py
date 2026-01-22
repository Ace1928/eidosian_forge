from . import _config
from .exceptions import FrozenAttributeError
def wrapped_pipe(instance, attrib, new_value):
    rv = new_value
    for setter in setters:
        rv = setter(instance, attrib, rv)
    return rv