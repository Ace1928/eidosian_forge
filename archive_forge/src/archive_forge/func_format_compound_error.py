import warnings
from .api import _, is_validator, FancyValidator, Invalid, NoDefault
from . import declarative
from .exc import FERuntimeWarning
def format_compound_error(v, indent=0):
    if isinstance(v, Exception):
        return str(v)
    if isinstance(v, dict):
        return ('%s\n' % (' ' * indent)).join(('%s: %s' % (k, format_compound_error(value, indent=len(k) + 2)) for k, value in sorted(v.items()) if value is not None))
    if isinstance(v, list):
        return ('%s\n' % (' ' * indent)).join(('%s' % format_compound_error(value, indent=indent) for value in v if value is not None))
    if isinstance(v, str):
        return v
    raise TypeError("I didn't expect something like %r" % v)