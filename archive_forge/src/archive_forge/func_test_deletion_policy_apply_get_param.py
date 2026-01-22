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
def test_deletion_policy_apply_get_param(self):
    tmpl = template_format.parse('\n        heat_template_version: 2016-04-08\n        parameters:\n          deletion_policy:\n            type: string\n            default: Delete\n        resources:\n          AResource:\n            type: ResourceWithPropsType\n            deletion_policy: {get_param: deletion_policy}\n            properties:\n              Foo: abc\n        ')
    self.stack = stack.Stack(self.ctx, 'stack_delpol_get_param', template.Template(tmpl))
    self.stack.validate()
    self.stack.store()
    self.stack.create()
    self.assertEqual((self.stack.CREATE, self.stack.COMPLETE), self.stack.state)