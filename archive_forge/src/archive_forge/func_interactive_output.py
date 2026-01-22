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
def interactive_output(f, controls):
    """Connect widget controls to a function.

    This function does not generate a user interface for the widgets (unlike `interact`).
    This enables customisation of the widget user interface layout.
    The user interface layout must be defined and displayed manually.
    """
    out = Output()

    def observer(change):
        kwargs = {k: v.value for k, v in controls.items()}
        show_inline_matplotlib_plots()
        with out:
            clear_output(wait=True)
            f(**kwargs)
            show_inline_matplotlib_plots()
    for k, w in controls.items():
        w.observe(observer, 'value')
    show_inline_matplotlib_plots()
    observer(None)
    return out