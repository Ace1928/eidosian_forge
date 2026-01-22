from unittest import mock
from keystoneauth1 import exceptions as kc_exceptions
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def test_user_token_err_noproject(self):
    stack_name = 'user_token_err_noprohect_stack'
    resource_name = 'user'
    t = template_format.parse(user_template)
    stack = utils.parse_stack(t, stack_name=stack_name)
    rsrc = stack[resource_name]
    ex = self.assertRaises(ValueError, rsrc._user_token)
    expected = "Can't get user token, user not yet created"
    self.assertEqual(expected, str(ex))