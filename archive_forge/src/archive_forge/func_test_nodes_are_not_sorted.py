from unittest import TestCase, SkipTest
def test_nodes_are_not_sorted(self):
    plot = hvnx.draw(self.g)
    assert all(self.nodes == plot.nodes.dimension_values(2))