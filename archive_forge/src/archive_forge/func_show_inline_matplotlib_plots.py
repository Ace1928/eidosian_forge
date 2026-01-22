from collections.abc import Iterable, Mapping
from inspect import signature, Parameter
from inspect import getcallargs
from inspect import getfullargspec as check_argspec
import sys
from IPython import get_ipython
from . import (Widget, ValueWidget, Text,
from IPython.display import display, clear_output
from traitlets import HasTraits, Any, Unicode, observe
from numbers import Real, Integral
from warnings import warn
def show_inline_matplotlib_plots():
    """Show matplotlib plots immediately if using the inline backend.

    With ipywidgets 6.0, matplotlib plots don't work well with interact when
    using the inline backend that comes with ipykernel. Basically, the inline
    backend only shows the plot after the entire cell executes, which does not
    play well with drawing plots inside of an interact function. See
    https://github.com/jupyter-widgets/ipywidgets/issues/1181/ and
    https://github.com/ipython/ipython/issues/10376 for more details. This
    function displays any matplotlib plots if the backend is the inline backend.
    """
    if 'matplotlib' not in sys.modules:
        return
    try:
        import matplotlib as mpl
        from ipykernel.pylab.backend_inline import flush_figures
    except ImportError:
        return
    if mpl.get_backend() == 'module://ipykernel.pylab.backend_inline' or mpl.get_backend() == 'module://matplotlib_inline.backend_inline':
        flush_figures()