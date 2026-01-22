import taskflow.engines as engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def test_multiply_array(self):
    expected = self.flow_store.copy()
    expected.update({'product': 120})
    requires = self.flow_store.keys()
    provides = 'product'
    flow = linear_flow.Flow('square array flow')
    flow.add(base.ReduceFunctorTask(multiply, requires=requires, provides=provides))
    result = engines.run(flow, store=self.flow_store)
    self.assertDictEqual(expected, result)