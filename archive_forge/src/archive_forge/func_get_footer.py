from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasCombine, CompositeCanvas
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, VAlign
from .container import WidgetContainerMixin
from .filler import Filler
from .widget import Widget, WidgetError
def get_footer(self) -> FooterWidget:
    warnings.warn(f'method `{self.__class__.__name__}.get_footer` is deprecated, standard property `{self.__class__.__name__}.footer` should be used instead', PendingDeprecationWarning, stacklevel=2)
    return self.footer