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
def test_load_parent_resource(self):
    self.stack = stack.Stack(self.ctx, 'load_parent_resource', self.tmpl, parent_resource='parent')
    self.stack.store()
    stk = stack_object.Stack.get_by_id(self.ctx, self.stack.id)
    t = template.Template.load(self.ctx, stk.raw_template_id)
    self.patchobject(template.Template, 'load', return_value=t)
    self.patchobject(stack.Stack, '__init__', return_value=None)
    stack.Stack.load(self.ctx, stack_id=self.stack.id)
    stack.Stack.__init__.assert_called_once_with(self.ctx, stk.name, t, stack_id=stk.id, action=stk.action, status=stk.status, status_reason=stk.status_reason, timeout_mins=stk.timeout, disable_rollback=stk.disable_rollback, parent_resource='parent', owner_id=None, stack_user_project_id=None, created_time=mock.ANY, updated_time=None, user_creds_id=stk.user_creds_id, tenant_id='test_tenant_id', use_stored_context=False, username=mock.ANY, convergence=False, current_traversal=self.stack.current_traversal, prev_raw_template_id=None, current_deps=None, cache_data=None, nested_depth=0, deleted_time=None, refresh_cred=False)
    template.Template.load.assert_called_once_with(self.ctx, stk.raw_template_id, stk.raw_template)