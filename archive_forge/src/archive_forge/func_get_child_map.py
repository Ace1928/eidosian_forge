import time
from . import debug, errors, osutils, revision, trace
def get_child_map(self, keys):
    """Get a mapping from parents to children of the specified keys.

        This is simply the inversion of get_parent_map.  Only supplied keys
        will be discovered as children.
        :return: a dict of key:child_list for keys.
        """
    parent_map = self._parents_provider.get_parent_map(keys)
    parent_child = {}
    for child, parents in sorted(parent_map.items()):
        for parent in parents:
            parent_child.setdefault(parent, []).append(child)
    return parent_child