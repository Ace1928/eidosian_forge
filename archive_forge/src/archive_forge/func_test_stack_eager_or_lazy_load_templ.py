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
def test_stack_eager_or_lazy_load_templ(self):
    self.stack = stack.Stack(self.ctx, 'test_stack_eager_or_lazy_tmpl', self.tmpl)
    self.stack.store()
    ctx1 = utils.dummy_context()
    s1_db_result = db_api.stack_get(ctx1, self.stack.id, eager_load=True)
    s1_obj = stack_object.Stack._from_db_object(ctx1, stack_object.Stack(), s1_db_result)
    self.assertIsNotNone(s1_obj._raw_template)
    self.assertIsNotNone(s1_obj.raw_template)
    ctx2 = utils.dummy_context()
    s2_db_result = db_api.stack_get(ctx2, self.stack.id, eager_load=False)
    s2_obj = stack_object.Stack._from_db_object(ctx2, stack_object.Stack(), s2_db_result)
    self.assertFalse(hasattr(s2_obj, '_raw_template'))
    self.assertIsNotNone(s2_obj.raw_template)
    self.assertIsNotNone(s2_obj._raw_template)