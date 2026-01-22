from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_collapse_nothing(self):
    d = {1: [2, 3], 2: [], 3: []}
    self.assertCollapsed(d, d)
    d = {1: [2], 2: [3, 4], 3: [5], 4: [5], 5: []}
    self.assertCollapsed(d, d)