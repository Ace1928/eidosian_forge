from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings
from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls
def mpl_runner(safe_execfile):
    """Factory to return a matplotlib-enabled runner for %run.

    Parameters
    ----------
    safe_execfile : function
        This must be a function with the same interface as the
        :meth:`safe_execfile` method of IPython.

    Returns
    -------
    A function suitable for use as the ``runner`` argument of the %run magic
    function.
    """

    def mpl_execfile(fname, *where, **kw):
        """matplotlib-aware wrapper around safe_execfile.

        Its interface is identical to that of the :func:`execfile` builtin.

        This is ultimately a call to execfile(), but wrapped in safeties to
        properly handle interactive rendering."""
        import matplotlib
        import matplotlib.pyplot as plt
        with matplotlib.rc_context({'interactive': False}):
            safe_execfile(fname, *where, **kw)
        if matplotlib.is_interactive():
            plt.show()
        if plt.draw_if_interactive.called:
            plt.draw()
            plt.draw_if_interactive.called = False
        try:
            da = plt.draw_all
        except AttributeError:
            pass
        else:
            da()
    return mpl_execfile