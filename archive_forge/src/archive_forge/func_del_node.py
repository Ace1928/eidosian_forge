import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def del_node(self, name, index=None):
    """Delete a node from the graph.

        Given a node's name all node(s) with that same name
        will be deleted if 'index' is not specified or set
        to None.
        If there are several nodes with that same name and
        'index' is given, only the node in that position
        will be deleted.

        'index' should be an integer specifying the position
        of the node to delete. If index is larger than the
        number of nodes with that name, no action is taken.

        If nodes are deleted it returns True. If no action
        is taken it returns False.
        """
    if isinstance(name, Node):
        name = name.get_name()
    if name in self.obj_dict['nodes']:
        if index is not None and index < len(self.obj_dict['nodes'][name]):
            del self.obj_dict['nodes'][name][index]
            return True
        else:
            del self.obj_dict['nodes'][name]
            return True
    return False