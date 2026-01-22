from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_json_list(self):
    schema = {'Type': 'Json'}
    val = ['fizz', 'buzz']
    p = new_parameter('p', schema, val)
    self.assertIsInstance(p.value(), list)
    self.assertIn('fizz', p.value())
    self.assertIn('buzz', p.value())