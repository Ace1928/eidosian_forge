from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
def test_handles_multiple_value_params(self):
    self.param_types = {'foo': util.PARAM_TYPE_MULTI}
    self.params.add('foo', 'foo value 2')
    result = util.get_allowed_params(self.params, self.param_types)
    self.assertEqual(2, len(result['foo']))
    self.assertIn('foo value', result['foo'])
    self.assertIn('foo value 2', result['foo'])