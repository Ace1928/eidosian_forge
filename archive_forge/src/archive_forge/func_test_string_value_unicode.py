from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_string_value_unicode(self):
    schema = {'Type': 'String'}
    p = new_parameter('p', schema, u'test♥')
    self.assertEqual(u'test♥', p.value())