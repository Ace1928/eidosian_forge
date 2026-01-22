from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_nested_prior_linear(self):
    r = lf.Flow('root')
    r.add(test_utils.TaskOneReturn('root.1'), test_utils.TaskOneReturn('root.2'))
    sub_r = lf.Flow('subroot')
    sub_r_1 = test_utils.TaskOneReturn('subroot.1')
    sub_r.add(sub_r_1)
    r.add(sub_r)
    c = compiler.PatternCompiler(r).compile()
    self.assertEqual([[], ['root.2', 'root.1']], _get_scopes(c, sub_r_1))