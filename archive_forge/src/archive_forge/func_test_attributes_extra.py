import collections
import json
import os
from unittest import mock
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources import template_resource
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_attributes_extra(self):
    provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Outputs': {'Foo': {'Value': 'bar'}, 'Blarg': {'Value': 'wibble'}}}
    files = {'test_resource.template': json.dumps(provider)}

    class DummyResource(generic_rsrc.GenericResource):
        attributes_schema = {'Foo': attributes.Schema('A test attribute')}
    env = environment.Environment()
    resource._register_class('DummyResource', DummyResource)
    env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
    stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
    definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource')
    temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
    self.assertIsNone(temp_res.validate())