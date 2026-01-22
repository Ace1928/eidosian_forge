import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def generate_root_texts(self, revs):
    """Generate VersionedFiles for all root ids.

        Args:
          revs: the revisions to include
        """
    from ..tsort import topo_sort
    graph = self.source.get_graph()
    parent_map = graph.get_parent_map(revs)
    rev_order = topo_sort(parent_map)
    rev_id_to_root_id = self._find_root_ids(revs, parent_map, graph)
    root_id_order = [(rev_id_to_root_id[rev_id], rev_id) for rev_id in rev_order]
    root_id_order.sort(key=operator.itemgetter(0))
    if len(revs) > self.known_graph_threshold:
        graph = self.source.get_known_graph_ancestry(revs)
    new_roots_stream = _new_root_data_stream(root_id_order, rev_id_to_root_id, parent_map, self.source, graph)
    return [('texts', new_roots_stream)]