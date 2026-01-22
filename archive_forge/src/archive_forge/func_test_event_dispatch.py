import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_event_dispatch(self):
    env = environment.Environment()
    evt = eventlet.event.Event()
    sink = fakes.FakeEventSink(evt)
    env.register_event_sink('dummy', lambda: sink)
    env.load({'event_sinks': [{'type': 'dummy'}]})
    stk = stack.Stack(self.ctx, 'test', template.Template(empty_template, env=env))
    stk.thread_group_mgr = service.ThreadGroupManager()
    self.addCleanup(stk.thread_group_mgr.stop, stk.id)
    stk.store()
    stk._add_event('CREATE', 'IN_PROGRESS', '')
    evt.wait()
    expected = [{'id': mock.ANY, 'timestamp': mock.ANY, 'type': 'os.heat.event', 'version': '0.1', 'payload': {'physical_resource_id': stk.id, 'resource_action': 'CREATE', 'resource_name': 'test', 'resource_properties': {}, 'resource_status': 'IN_PROGRESS', 'resource_status_reason': '', 'resource_type': 'OS::Heat::Stack', 'stack_id': stk.id, 'version': '0.1'}}]
    self.assertEqual(expected, sink.events)