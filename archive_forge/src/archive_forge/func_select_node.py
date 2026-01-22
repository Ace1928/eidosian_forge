from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def select_node(self, node):
    """ Selects a node.

        It is called by the controller when it selects a node and can be
        called from the outside to select a node directly. The derived widget
        should overwrite this method and change the node state to selected
        when called.

        :Parameters:
            `node`
                The node to be selected.

        :Returns:
            bool, True if the node was selected, False otherwise.

        .. warning::

            This method must be called by the derived widget using super if it
            is overwritten.
        """
    nodes = self.selected_nodes
    if node in nodes:
        return False
    if not self.multiselect and len(nodes):
        self.clear_selection()
    if node not in nodes:
        nodes.append(node)
    self._anchor = node
    self._last_selected_node = node
    return True