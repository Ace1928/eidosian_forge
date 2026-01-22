from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_graph_nested_graph(self):
    a, b, c, d, e, f, g = test_utils.make_many(7)
    flo = gf.Flow('test')
    flo.add(a, b, c, d)
    flo2 = gf.Flow('test2')
    flo2.add(e, f, g)
    flo.add(flo2)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(11, len(g))
    self.assertCountEqual(g.edges(), [('test', 'a'), ('test', 'b'), ('test', 'c'), ('test', 'd'), ('test', 'test2'), ('test2', 'e'), ('test2', 'f'), ('test2', 'g'), ('e', 'test2[$]'), ('f', 'test2[$]'), ('g', 'test2[$]'), ('test2[$]', 'test[$]'), ('a', 'test[$]'), ('b', 'test[$]'), ('c', 'test[$]'), ('d', 'test[$]')])