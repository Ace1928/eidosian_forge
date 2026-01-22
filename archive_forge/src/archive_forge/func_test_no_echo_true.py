from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_no_echo_true(self):
    p = new_parameter('anechoic', {'Type': self.p_type, 'NoEcho': 'true'}, self.value)
    self.assertTrue(p.hidden())
    self.assertEqual('******', str(p))