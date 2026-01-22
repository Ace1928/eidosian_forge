from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def get_selectable_nodes(self):
    """(internal) Returns a list of the nodes that can be selected. It can
        be overwritten by the derived widget to return the correct list.

        This list is used to determine which nodes to select with group
        selection. E.g. the last element in the list will be selected when
        home is pressed, pagedown will move (or add to, if shift is held) the
        selection from the current position by negative :attr:`page_count`
        nodes starting from the position of the currently selected node in
        this list and so on. Still, nodes can be selected even if they are not
        in this list.

        .. note::

            It is safe to dynamically change this list including removing,
            adding, or re-arranging its elements. Nodes can be selected even
            if they are not on this list. And selected nodes removed from the
            list will remain selected until :meth:`deselect_node` is called.

        .. warning::

            Layouts display their children in the reverse order. That is, the
            contents of :attr:`~kivy.uix.widget.Widget.children` is displayed
            form right to left, bottom to top. Therefore, internally, the
            indices of the elements returned by this function are reversed to
            make it work by default for most layouts so that the final result
            is consistent e.g. home, although it will select the last element
            in this list visually, will select the first element when
            counting from top to bottom and left to right. If this behavior is
            not desired, a reversed list should be returned instead.

        Defaults to returning :attr:`~kivy.uix.widget.Widget.children`.
        """
    return self.children