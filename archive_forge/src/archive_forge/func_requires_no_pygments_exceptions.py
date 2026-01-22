import pytest
from mako.ext.beaker_cache import has_beaker
from mako.util import update_wrapper
def requires_no_pygments_exceptions(fn):

    def go(*arg, **kw):
        from mako import exceptions
        exceptions._install_fallback()
        try:
            return fn(*arg, **kw)
        finally:
            exceptions._install_highlighting()
    return update_wrapper(go, fn)