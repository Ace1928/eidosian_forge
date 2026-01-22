from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
def test_bogus_param_type(self):
    self.param_types = {'foo': 'blah'}
    self.assertRaises(AssertionError, util.get_allowed_params, self.params, self.param_types)