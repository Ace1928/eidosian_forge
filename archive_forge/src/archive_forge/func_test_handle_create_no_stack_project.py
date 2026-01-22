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
def test_handle_create_no_stack_project(self):
    stack_name = 'stackuser_crnoprj'
    resource_name = 'user'
    project_id = 'aproject123'
    user_id = 'auser123'
    rsrc = self._user_create(stack_name=stack_name, project_id=project_id, user_id=user_id)
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    rs_data = resource_data_object.ResourceData.get_all(rsrc)
    self.assertEqual({'user_id': user_id}, rs_data)
    self.fc.create_stack_domain_project.assert_called_once_with(self.stack.id)
    expected_username = '%s-%s-%s' % (stack_name, resource_name, 'aabbcc')
    self.fc.create_stack_domain_user.assert_called_once_with(password=None, project_id=project_id, username=expected_username)