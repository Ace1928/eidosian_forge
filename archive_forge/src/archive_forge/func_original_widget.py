from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from .widget import Widget, WidgetError, WidgetWarning, delegate_to_widget_mixin
@original_widget.setter
def original_widget(self, original_widget: WrappedWidget) -> None:
    self._original_widget = original_widget
    self._invalidate()