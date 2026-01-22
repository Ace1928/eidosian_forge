from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_unordered_nested_in_linear(self):
    a, b, c, d = test_utils.make_many(4)
    inner_flo = uf.Flow('ut').add(b, c)
    flo = lf.Flow('lt').add(a, inner_flo, d)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(8, len(g))
    self.assertCountEqual(g.edges(), [('lt', 'a'), ('a', 'ut'), ('ut', 'b'), ('ut', 'c'), ('b', 'ut[$]'), ('c', 'ut[$]'), ('ut[$]', 'd'), ('d', 'lt[$]')])