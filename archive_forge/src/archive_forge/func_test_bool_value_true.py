from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_bool_value_true(self):
    schema = {'Type': 'Boolean'}
    for val in ('1', 't', 'true', 'on', 'y', 'yes', True, 1):
        bo = new_parameter('bo', schema, val)
        self.assertTrue(bo.value())