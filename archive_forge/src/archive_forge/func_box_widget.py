from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from .constants import Sizing
from .widget_decoration import WidgetDecoration, WidgetError
@box_widget.setter
def box_widget(self, widget: WrappedWidget) -> None:
    warnings.warn('original stored as original_widget, keep for compatibility', PendingDeprecationWarning, stacklevel=2)
    self.original_widget = widget