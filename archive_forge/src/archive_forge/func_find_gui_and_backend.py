from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings
from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls
def find_gui_and_backend(gui=None, gui_select=None):
    """Given a gui string return the gui and mpl backend.

    Parameters
    ----------
    gui : str
        Can be one of ('tk','gtk','wx','qt','qt4','inline','agg').
    gui_select : str
        Can be one of ('tk','gtk','wx','qt','qt4','inline').
        This is any gui already selected by the shell.

    Returns
    -------
    A tuple of (gui, backend) where backend is one of ('TkAgg','GTKAgg',
    'WXAgg','Qt4Agg','module://matplotlib_inline.backend_inline','agg').
    """
    import matplotlib
    has_unified_qt_backend = getattr(matplotlib, '__version_info__', (0, 0)) >= (3, 5)
    backends_ = dict(backends)
    if not has_unified_qt_backend:
        backends_['qt'] = 'qt5agg'
    if gui and gui != 'auto':
        backend = backends_[gui]
        if gui == 'agg':
            gui = None
    else:
        backend = matplotlib.rcParamsOrig['backend']
        gui = backend2gui.get(backend, None)
        if gui_select and gui != gui_select:
            gui = gui_select
            backend = backends_[gui]
    return (gui, backend)