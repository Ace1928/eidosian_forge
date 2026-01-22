import taskflow.engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def test_it_runs(self):
    values = []
    bof = BunchOfFunctions(values)
    t = base.FunctorTask
    flow = linear_flow.Flow('test')
    flow.add(t(bof.run_one, revert=bof.revert_one), t(bof.run_fail))
    self.assertRaisesRegex(RuntimeError, '^Woot', taskflow.engines.run, flow)
    self.assertEqual(['one', 'fail', 'revert one'], values)