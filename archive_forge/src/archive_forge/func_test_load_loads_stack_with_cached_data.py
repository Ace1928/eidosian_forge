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
@mock.patch.object(parser.Stack, 'load')
@mock.patch.object(resource.Resource, '_load_data')
@mock.patch.object(template.Template, 'load')
def test_load_loads_stack_with_cached_data(self, mock_tmpl_load, mock_load_data, mock_stack_load):
    tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'res': {'type': 'GenericResourceType'}}}, env=self.env)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    stack.store()
    mock_tmpl_load.return_value = tmpl
    res = stack['res']
    res.current_template_id = stack.t.id
    res.store()
    data = {'bar': {'atrr1': 'baz', 'attr2': 'baz2'}}
    mock_stack_load.return_value = stack
    resource.Resource.load(stack.context, res.id, stack.current_traversal, True, data)
    self.assertTrue(mock_stack_load.called)
    mock_stack_load.assert_called_with(stack.context, stack_id=stack.id, cache_data=data)
    self.assertTrue(mock_load_data.called)