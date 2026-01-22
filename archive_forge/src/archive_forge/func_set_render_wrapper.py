import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Union
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
from ..util import find_available_port, is_in_jupyter
from .render import DependencyRenderer, EntityRenderer, SpanRenderer
def set_render_wrapper(func: Callable[[str], str]) -> None:
    """Set an optional wrapper function that is called around the generated
    HTML markup on displacy.render. This can be used to allow integration into
    other platforms, similar to Jupyter Notebooks that require functions to be
    called around the HTML. It can also be used to implement custom callbacks
    on render, or to embed the visualization in a custom page.

    func (callable): Function to call around markup before rendering it. Needs
        to take one argument, the HTML markup, and should return the desired
        output of displacy.render.
    """
    global RENDER_WRAPPER
    if not hasattr(func, '__call__'):
        raise ValueError(Errors.E110.format(obj=type(func)))
    RENDER_WRAPPER = func