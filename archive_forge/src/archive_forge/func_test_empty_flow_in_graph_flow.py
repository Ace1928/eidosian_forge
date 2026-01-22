from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_empty_flow_in_graph_flow(self):
    flow = lf.Flow('lf')
    a = test_utils.ProvidesRequiresTask('a', provides=['a'], requires=[])
    b = test_utils.ProvidesRequiresTask('b', provides=[], requires=['a'])
    empty_flow = lf.Flow('empty')
    flow.add(a, empty_flow, b)
    compilation = compiler.PatternCompiler(flow).compile()
    g = compilation.execution_graph
    self.assertTrue(g.has_edge(flow, a))
    self.assertTrue(g.has_edge(a, empty_flow))
    empty_flow_successors = list(g.successors(empty_flow))
    self.assertEqual(1, len(empty_flow_successors))
    empty_flow_terminal = empty_flow_successors[0]
    self.assertIs(empty_flow, empty_flow_terminal.flow)
    self.assertEqual(compiler.FLOW_END, g.nodes[empty_flow_terminal]['kind'])
    self.assertTrue(g.has_edge(empty_flow_terminal, b))