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
def test_user_token_err_nopassword(self):
    stack_name = 'stackuser_testtoken_err_nopwd'
    project_id = 'aproject123'
    user_id = 'auser123'
    rsrc = self._user_create(stack_name=stack_name, project_id=project_id, user_id=user_id)
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    ex = self.assertRaises(ValueError, rsrc._user_token)
    expected = "Can't get user token without password"
    self.assertEqual(expected, str(ex))
    self.fc.stack_domain_user_token.assert_not_called()