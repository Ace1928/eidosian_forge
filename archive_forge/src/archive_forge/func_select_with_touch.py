from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def select_with_touch(self, node, touch=None):
    """(internal) Processes a touch on the node. This should be called by
        the derived widget when a node is touched and is to be used for
        selection. Depending on the keyboard keys pressed and the
        configuration, it could select or deslect this and other nodes in the
        selectable nodes list, :meth:`get_selectable_nodes`.

        :Parameters:
            `node`
                The node that received the touch. Can be None for a scroll
                type touch.
            `touch`
                Optionally, the touch. Defaults to None.

        :Returns:
            bool, True if the touch was used, False otherwise.
        """
    multi = self.multiselect
    multiselect = multi and (self._ctrl_down or self.touch_multiselect)
    range_select = multi and self._shift_down
    if touch and 'button' in touch.profile and (touch.button in ('scrollup', 'scrolldown', 'scrollleft', 'scrollright')):
        node_src, idx_src = self._resolve_last_node()
        node, idx = self.goto_node(touch.button, node_src, idx_src)
        if node == node_src:
            return False
        if range_select:
            self._select_range(multiselect, True, node, idx)
        else:
            if not multiselect:
                self.clear_selection()
            self.select_node(node)
        return True
    if node is None:
        return False
    if node in self.selected_nodes and (not range_select):
        if multiselect:
            self.deselect_node(node)
        else:
            selected_node_count = len(self.selected_nodes)
            self.clear_selection()
            if not self.touch_deselect_last or selected_node_count > 1:
                self.select_node(node)
    elif range_select:
        self._select_range(multiselect, not multiselect, node, 0)
    else:
        if not multiselect:
            self.clear_selection()
        self.select_node(node)
    return True