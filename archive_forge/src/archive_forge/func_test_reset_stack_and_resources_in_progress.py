from unittest import mock
import uuid
import eventlet.queue
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import messaging
from heat.common import service_utils
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine import environment
from heat.engine import resource
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_reset_stack_and_resources_in_progress(self):

    def mock_stack_resource(name, action, status):
        rs = mock.MagicMock()
        rs.name = name
        rs.action = action
        rs.status = status
        rs.IN_PROGRESS = 'IN_PROGRESS'
        rs.FAILED = 'FAILED'

        def mock_resource_state_set(a, s, reason='engine_down'):
            rs.status = s
            rs.action = a
            rs.status_reason = reason
        rs.state_set = mock_resource_state_set
        return rs
    stk_name = 'test_stack'
    stk = tools.get_stack(stk_name, self.ctx)
    stk.action = 'CREATE'
    stk.status = 'IN_PROGRESS'
    resources = {'r1': mock_stack_resource('r1', 'UPDATE', 'COMPLETE'), 'r2': mock_stack_resource('r2', 'UPDATE', 'IN_PROGRESS'), 'r3': mock_stack_resource('r3', 'UPDATE', 'FAILED')}
    stk._resources = resources
    reason = 'Test resetting stack and resources in progress'
    stk.reset_stack_and_resources_in_progress(reason)
    self.assertEqual('FAILED', stk.status)
    self.assertEqual('COMPLETE', stk.resources.get('r1').status)
    self.assertEqual('FAILED', stk.resources.get('r2').status)
    self.assertEqual('FAILED', stk.resources.get('r3').status)