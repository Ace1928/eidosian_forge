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
def test_stack_update_preview_replaced_type(self):
    new_tmpl = self.old_tmpl.replace('OS::Nova::Server', 'OS::Heat::None')
    result = self._test_stack_update_preview(self.old_tmpl, new_tmpl)
    replaced = [x for x in result['replaced']][0]
    self.assertEqual('web_server', replaced['resource_name'])
    empty_sections = ('added', 'deleted', 'unchanged', 'updated')
    for section in empty_sections:
        section_contents = [x for x in result[section]]
        self.assertEqual([], section_contents)