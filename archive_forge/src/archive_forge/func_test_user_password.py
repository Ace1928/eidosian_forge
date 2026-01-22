from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import node_data
from heat.engine.resources.aws.iam import user
from heat.engine.resources.openstack.heat import access_policy as ap
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def test_user_password(self):
    t = template_format.parse(user_template_password)
    stack = utils.parse_stack(t, stack_name=self.stack_name)
    project_id = 'stackproject'
    password = u'myP@ssW0rd'
    rsrc = self.create_user(t, stack, 'CfnUser', project_id=project_id, password=password)
    self.assertEqual('dummy_user', rsrc.resource_id)
    self.assertEqual(self.username, rsrc.FnGetRefId())
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.mock_create_project.assert_called_once_with(stack.id)
    self.mock_create_user.assert_called_once_with(password=password, project_id=project_id, username=self.username)