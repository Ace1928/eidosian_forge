import time
from . import debug, errors, osutils, revision, trace
def find_descendants(self, old_key, new_key):
    """Find descendants of old_key that are ancestors of new_key."""
    child_map = self.get_child_map(self._find_descendant_ancestors(old_key, new_key))
    graph = Graph(DictParentsProvider(child_map))
    searcher = graph._make_breadth_first_searcher([old_key])
    list(searcher)
    return searcher.seen