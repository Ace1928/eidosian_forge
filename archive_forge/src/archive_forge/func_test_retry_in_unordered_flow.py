from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_retry_in_unordered_flow(self):
    flo = uf.Flow('test', retry.AlwaysRevert('c'))
    compilation = compiler.PatternCompiler(flo).compile()
    self.assertEqual(3, len(compilation.execution_graph))
    self.assertEqual(2, compilation.execution_graph.number_of_edges())