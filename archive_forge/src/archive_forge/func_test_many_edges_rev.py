from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_many_edges_rev(self):
    self._dep_test_rev(('last', 'e1'), ('last', 'mid1'), ('last', 'mid2'), ('mid1', 'e2'), ('mid1', 'mid3'), ('mid2', 'mid3'), ('mid3', 'e3'))