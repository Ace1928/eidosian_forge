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
@mock.patch.object(resource.Resource, 'update_template_diff_properties')
@mock.patch.object(resource.Resource, '_needs_update')
def test_update_convergence(self, mock_nu, mock_utd):
    tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'test_res': {'Type': 'ResourceWithPropsType'}}}, env=self.env)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, convergence=True)
    stack.thread_group_mgr = tools.DummyThreadGroupManager()
    stack.converge_stack(stack.t, action=stack.CREATE)
    res = stack.resources['test_res']
    res.requires = {2}
    res.action = res.CREATE
    res.store()
    self._assert_resource_lock(res.id, None, None)
    new_temp = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'test_res': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'abc'}}}}, env=self.env)
    new_temp.store(stack.context)
    new_stack = parser.Stack(utils.dummy_context(), 'test_stack', new_temp, stack_id=self.stack.id, convergence=True)
    tr = scheduler.TaskRunner(res.update_convergence, new_temp.id, {4, 3}, 'engine-007', 120, new_stack)
    tr()
    self.assertCountEqual([3, 4], res.requires)
    self.assertEqual(res.action, resource.Resource.UPDATE)
    self.assertEqual(res.status, resource.Resource.COMPLETE)
    self._assert_resource_lock(res.id, None, 2)