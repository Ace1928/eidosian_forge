from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_invalid_decider_depth(self):
    g_1 = utils.ProgressingTask(name='g-1')
    g_2 = utils.ProgressingTask(name='g-2')
    for not_a_depth in ['not-a-depth', object(), 2, 3.4, False]:
        flow = gf.Flow('g')
        flow.add(g_1, g_2)
        self.assertRaises((ValueError, TypeError), flow.link, g_1, g_2, decider=lambda history: False, decider_depth=not_a_depth)