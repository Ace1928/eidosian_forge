import functools
import matplotlib as mpl
from cycler import cycler
from . import palettes
def reset_orig():
    """Restore all RC params to original settings (respects custom rc)."""
    from . import _orig_rc_params
    mpl.rcParams.update(_orig_rc_params)