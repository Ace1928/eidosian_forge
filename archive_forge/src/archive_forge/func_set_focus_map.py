from __future__ import annotations
import typing
from collections.abc import Hashable, Mapping
from urwid.canvas import CompositeCanvas
from .widget import WidgetError, delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
def set_focus_map(self, focus_map: dict[Hashable | None, Hashable] | None) -> None:
    """
        Set the focus attribute mapping dictionary
        {from_attr: to_attr, ...}

        If None this widget will use the attr mapping instead (no change
        when in focus).

        Note this function does not accept a single attribute the way the
        constructor does.  You must specify {None: attribute} instead.

        >>> from urwid import Text
        >>> w = AttrMap(Text(u"hi"), {})
        >>> w.set_focus_map({'a':'b'})
        >>> w
        <AttrMap fixed/flow widget <Text fixed/flow widget 'hi'> attr_map={} focus_map={'a': 'b'}>
        >>> w.set_focus_map(None)
        >>> w
        <AttrMap fixed/flow widget <Text fixed/flow widget 'hi'> attr_map={}>
        """
    if focus_map is not None:
        for from_attr, to_attr in focus_map.items():
            if not isinstance(from_attr, Hashable) or not isinstance(to_attr, Hashable):
                raise AttrMapError(f'{from_attr!r}:{to_attr!r} attribute mapping is invalid. Attributes must be hashable')
    self._focus_map = focus_map
    self._invalidate()