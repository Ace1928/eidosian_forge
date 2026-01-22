from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def make_breaking_graph(self, ancestors, break_on):
    """Make a Graph that raises an exception if we hit a node."""
    g = self.make_graph(ancestors)
    orig_parent_map = g.get_parent_map

    def get_parent_map(keys):
        bad_keys = set(keys).intersection(break_on)
        if bad_keys:
            self.fail('key(s) {} was accessed'.format(sorted(bad_keys)))
        return orig_parent_map(keys)
    g.get_parent_map = get_parent_map
    return g