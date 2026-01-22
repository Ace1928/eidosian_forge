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
def test_resource_class_to_hot_template(self):

    class TestResource(resource.Resource):
        list_schema = {'wont_show_up': {'Type': 'Number'}}
        map_schema = {'will_show_up': {'Type': 'Integer'}}
        properties_schema = {'name': {'Type': 'String'}, 'bool': {'Type': 'Boolean'}, 'implemented': {'Type': 'String', 'Implemented': True, 'AllowedPattern': '.*', 'MaxLength': 7, 'MinLength': 2, 'Required': True}, 'not_implemented': {'Type': 'String', 'Implemented': False}, 'number': {'Type': 'Number', 'MaxValue': 77, 'MinValue': 41, 'Default': 42}, 'list': {'Type': 'List', 'Schema': {'Type': 'Map', 'Schema': list_schema}}, 'map': {'Type': 'Map', 'Schema': map_schema}, 'hidden': properties.Schema(properties.Schema.STRING, support_status=support.SupportStatus(status=support.HIDDEN))}
        attributes_schema = {'output1': attributes.Schema('output1_desc'), 'output2': attributes.Schema('output2_desc')}
    expected_template = {'heat_template_version': '2016-10-14', 'description': 'Initial template of TestResource', 'parameters': {'name': {'type': 'string'}, 'bool': {'type': 'boolean'}, 'implemented': {'type': 'string', 'constraints': [{'length': {'max': 7, 'min': 2}}, {'allowed_pattern': '.*'}]}, 'number': {'type': 'number', 'constraints': [{'range': {'max': 77, 'min': 41}}], 'default': 42}, 'list': {'type': 'comma_delimited_list'}, 'map': {'type': 'json'}}, 'resources': {'TestResource': {'type': 'Test::Resource::resource', 'properties': {'name': {'get_param': 'name'}, 'bool': {'get_param': 'bool'}, 'implemented': {'get_param': 'implemented'}, 'number': {'get_param': 'number'}, 'list': {'get_param': 'list'}, 'map': {'get_param': 'map'}}}}, 'outputs': {'output1': {'description': 'output1_desc', 'value': {'get_attr': ['TestResource', 'output1']}}, 'output2': {'description': 'output2_desc', 'value': {'get_attr': ['TestResource', 'output2']}}, 'show': {'description': u'Detailed information about resource.', 'value': {'get_attr': ['TestResource', 'show']}}, 'OS::stack_id': {'value': {'get_resource': 'TestResource'}}}}
    self.assertEqual(expected_template, TestResource.resource_to_template('Test::Resource::resource', template_type='hot'))