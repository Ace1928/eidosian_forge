from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_task(self):
    task = test_utils.DummyTask(name='a')
    g = _replicate_graph_with_names(compiler.PatternCompiler(task).compile())
    self.assertEqual(['a'], list(g.nodes()))
    self.assertEqual([], list(g.edges()))