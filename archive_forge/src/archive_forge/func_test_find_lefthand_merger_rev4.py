from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_find_lefthand_merger_rev4(self):
    self.check_merger(None, ancestry_1, b'rev4', b'rev2a')