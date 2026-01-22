from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def leaf_nodes(self, app=None):
    """
        Return all leaf nodes - that is, nodes with no dependents in their app.
        These are the "most current" version of an app's schema.
        Having more than one per app is technically an error, but one that
        gets handled further up, in the interactive command - it's usually the
        result of a VCS merge and needs some user input.
        """
    leaves = set()
    for node in self.nodes:
        if all((key[0] != node[0] for key in self.node_map[node].children)) and (not app or app == node[0]):
            leaves.add(node)
    return sorted(leaves)