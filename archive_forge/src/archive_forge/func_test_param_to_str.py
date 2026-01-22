from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_param_to_str(self):
    p = new_parameter('p', {'Type': self.p_type}, self.value)
    if self.p_type == 'Json':
        self.assertEqual(json.loads(self.expected), json.loads(str(p)))
    else:
        self.assertEqual(self.expected, str(p))