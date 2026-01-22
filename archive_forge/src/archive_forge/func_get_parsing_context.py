from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def get_parsing_context():
    """returns the default dictionary for parsing strings in chempy"""
    import chempy
    from chempy.kinetics import rates
    from chempy.units import default_units, default_constants, to_unitless
    globals_ = dict(to_unitless=to_unitless, chempy=chempy)

    def _update(mod, keys=None):
        if keys is None:
            keys = dir(mod)
        globals_.update({k: getattr(mod, k) for k in keys if not k.startswith('_')})
    try:
        import numpy
    except ImportError:

        def _numpy_not_installed_raise(*args, **kwargs):
            raise ImportError('numpy not installed, no such method')

        class numpy:
            array = staticmethod(_numpy_not_installed_raise)
            log = staticmethod(_numpy_not_installed_raise)
            exp = staticmethod(_numpy_not_installed_raise)
    _update(numpy, keys='array log exp'.split())
    _update(rates)
    _update(chempy)
    for df in [default_units, default_constants]:
        if df is not None:
            globals_.update(df.as_dict())
    return globals_