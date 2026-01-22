from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_string_len_good(self):
    schema = {'Type': 'String', 'MinLength': '3', 'MaxLength': '3'}
    p = new_parameter('p', schema, 'foo')
    self.assertEqual('foo', p.value())