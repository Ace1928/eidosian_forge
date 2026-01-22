from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_starts_as_empty(self):
    f = gf.Flow('test')
    self.assertEqual(0, len(f))
    self.assertEqual([], list(f))
    self.assertEqual([], list(f.iter_links()))
    self.assertEqual(set(), f.requires)
    self.assertEqual(set(), f.provides)