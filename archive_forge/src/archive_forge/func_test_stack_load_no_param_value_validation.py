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
def test_stack_load_no_param_value_validation(self):
    """Test stack loading with disabled parameter value validation."""
    tmpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            flavor:\n                type: string\n                description: A flavor.\n                constraints:\n                    - custom_constraint: nova.flavor\n        resources:\n            a_resource:\n                type: GenericResourceType\n        ')
    fc = fakes.FakeClient()
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=fc)
    fc.flavors = mock.Mock()
    flavor = collections.namedtuple('Flavor', ['id', 'name'])
    flavor.id = '1234'
    flavor.name = 'dummy'
    fc.flavors.get.return_value = flavor
    test_env = environment.Environment({'flavor': '1234'})
    self.stack = stack.Stack(self.ctx, 'stack_with_custom_constraint', template.Template(tmpl, env=test_env))
    self.stack.validate()
    self.stack.store()
    self.stack.create()
    stack_id = self.stack.id
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    loaded_stack = stack.Stack.load(self.ctx, stack_id=self.stack.id)
    self.assertEqual(stack_id, loaded_stack.parameters['OS::stack_id'])
    fc.flavors.get.assert_called_once_with('1234')