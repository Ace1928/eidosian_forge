import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
def test_handle_delete_no_resource_id(self):
    self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template))
    self.stack.store()
    snippet = rsrc_defn.ResourceDefinition('aresource', 'OS::Heat::None')
    res = resource.Resource('aresource', snippet, self.stack)
    FakeClient = collections.namedtuple('Client', ['entity'])
    client = FakeClient(collections.namedtuple('entity', ['delete']))
    self.patchobject(resource.Resource, 'client', return_value=client)
    delete = mock.Mock()
    res.client().entity.delete = delete
    res.entity = 'entity'
    res.default_client_name = 'something'
    res.resource_id = None
    self.assertIsNone(res.handle_delete())
    self.assertEqual(0, delete.call_count)