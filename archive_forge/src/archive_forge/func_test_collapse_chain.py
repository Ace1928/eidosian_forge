from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_collapse_chain(self):
    d = {1: [2], 2: [3], 3: [4], 4: [5], 5: []}
    self.assertCollapsed({1: [5], 5: []}, d)
    d = {5: [4], 4: [3], 3: [2], 2: [1], 1: []}
    self.assertCollapsed({5: [1], 1: []}, d)
    d = {5: [3], 3: [4], 4: [1], 1: [2], 2: []}
    self.assertCollapsed({5: [2], 2: []}, d)