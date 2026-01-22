from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def iterate_open_nodes(self, node=None):
    """Generator to iterate over all the expended nodes starting from
        `node` and down. If `node` is `None`, the generator start with
        :attr:`root`.

        To get all the open nodes::

            treeview = TreeView()
            # ... add nodes ...
            for node in treeview.iterate_open_nodes():
                print(node)

        """
    if not node:
        node = self.root
    if self.hide_root and node is self.root:
        pass
    else:
        yield node
    if not node.is_open:
        return
    f = self.iterate_open_nodes
    for cnode in node.nodes:
        for ynode in f(cnode):
            yield ynode