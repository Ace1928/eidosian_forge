from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def get_shared_provider(self, info, ancestry, has_cached):
    pp = _mod_graph.DictParentsProvider(ancestry)
    if has_cached:
        pp.get_cached_parent_map = pp.get_parent_map
    return SharedInstrumentedParentsProvider(pp, self.calls, info)