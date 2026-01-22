from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def clear_selection(self):
    """ Deselects all the currently selected nodes.
        """
    deselect = self.deselect_node
    nodes = self.selected_nodes
    for node in nodes[:]:
        deselect(node)