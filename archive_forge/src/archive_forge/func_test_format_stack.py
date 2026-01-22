import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def test_format_stack(self):
    self.stack.created_time = datetime(1970, 1, 1)
    info = api.format_stack(self.stack)
    aws_id = 'arn:openstack:heat::test_tenant_id:stacks/test_stack/' + self.stack.id
    expected_stack_info = {'capabilities': [], 'creation_time': '1970-01-01T00:00:00Z', 'deletion_time': None, 'description': 'No description', 'disable_rollback': True, 'notification_topics': [], 'stack_action': 'CREATE', 'stack_name': 'test_stack', 'stack_owner': 'test_username', 'stack_status': 'IN_PROGRESS', 'stack_status_reason': '', 'stack_user_project_id': None, 'outputs': [], 'template_description': 'No description', 'timeout_mins': None, 'tags': [], 'parameters': {'AWS::Region': 'ap-southeast-1', 'AWS::StackId': aws_id, 'AWS::StackName': 'test_stack'}, 'stack_identity': {'path': '', 'stack_id': self.stack.id, 'stack_name': 'test_stack', 'tenant': 'test_tenant_id'}, 'updated_time': None, 'parent': None}
    self.assertEqual(expected_stack_info, info)