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
def test_get_template_resource_class(self):
    test_templ_name = 'file:///etc/heatr/frodo.yaml'
    minimal_temp = json.dumps({'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {}, 'Resources': {}})
    mock_get = self.patchobject(urlfetch, 'get', return_value=minimal_temp)
    env_str = {'resource_registry': {'resources': {'fred': {'OS::ResourceType': test_templ_name}}}}
    global_env = environment.Environment({}, user_env=False)
    global_env.load(env_str)
    with mock.patch('heat.engine.resources._environment', global_env):
        env = environment.Environment({})
    cls = env.get_class('OS::ResourceType', 'fred')
    self.assertNotEqual(template_resource.TemplateResource, cls)
    self.assertTrue(issubclass(cls, template_resource.TemplateResource))
    self.assertTrue(hasattr(cls, 'properties_schema'))
    self.assertTrue(hasattr(cls, 'attributes_schema'))
    mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('file',))