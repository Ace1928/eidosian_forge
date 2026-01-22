from __future__ import annotations
import typing
import warnings
from .attr_map import AttrMap
def set_focus_attr(self, focus_attr: Hashable) -> None:
    """
        Set the attribute to apply to the wapped widget when it is in
        focus

        If None this widget will use the attr instead (no change when in
        focus).

        >> w = AttrWrap(Divider("-"), 'old')
        >> w.set_focus_attr('new_attr')
        >> w
        <AttrWrap flow widget <Divider flow widget '-'> attr='old' focus_attr='new_attr'>
        >> w.set_focus_attr(None)
        >> w
        <AttrWrap flow widget <Divider flow widget '-'> attr='old'>
        """
    self.set_focus_map({None: focus_attr})