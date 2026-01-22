import taskflow.engines as engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def test_double_array(self):
    expected = self.flow_store.copy()
    expected.update({'double_a': 2, 'double_b': 4, 'double_c': 6, 'double_d': 8, 'double_e': 10})
    requires = self.flow_store.keys()
    provides = ['double_%s' % k for k in requires]
    flow = linear_flow.Flow('double array flow')
    flow.add(base.MapFunctorTask(double, requires=requires, provides=provides))
    result = engines.run(flow, store=self.flow_store)
    self.assertDictEqual(expected, result)