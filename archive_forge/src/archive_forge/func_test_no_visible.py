from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_no_visible(self):
    r = uf.Flow('root')
    atoms = []
    for i in range(0, 10):
        atoms.append(test_utils.TaskOneReturn('root.%s' % i))
    r.add(*atoms)
    c = compiler.PatternCompiler(r).compile()
    for a in atoms:
        self.assertEqual([], _get_scopes(c, a))