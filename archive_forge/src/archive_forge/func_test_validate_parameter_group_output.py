from unittest import mock
from oslo_messaging.rpc import dispatcher
import webob
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine.clients.os import glance
from heat.engine import environment
from heat.engine.hot import template as hot_tmpl
from heat.engine import resources
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_validate_parameter_group_output(self):
    engine = service.EngineService('a', 't')
    params = {'resource_registry': {'OS::Test::TestResource': 'https://server.test/nested.template'}}
    root_template_str = '\nheat_template_version: 2015-10-15\nparameters:\n    test_root_param:\n        type: string\nparameter_groups:\n-   label: RootTest\n    parameters:\n    -   test_root_param\nresources:\n    Nested:\n        type: OS::Test::TestResource\n'
    nested_template_str = '\nheat_template_version: 2015-10-15\nparameters:\n    test_param:\n        type: string\nparameter_groups:\n-   label: Test\n    parameters:\n    -   test_param\n'
    root_template = template_format.parse(root_template_str)
    self.patchobject(urlfetch, 'get')
    urlfetch.get.return_value = nested_template_str
    res = dict(engine.validate_template(self.ctx, root_template, params, show_nested=True))
    expected = {'Description': 'No description', 'ParameterGroups': [{'label': 'RootTest', 'parameters': ['test_root_param']}], 'Parameters': {'test_root_param': {'Description': '', 'Label': 'test_root_param', 'NoEcho': 'false', 'Type': 'String'}}, 'NestedParameters': {'Nested': {'Description': 'No description', 'ParameterGroups': [{'label': 'Test', 'parameters': ['test_param']}], 'Parameters': {'test_param': {'Description': '', 'Label': 'test_param', 'NoEcho': 'false', 'Type': 'String'}}, 'Type': 'OS::Test::TestResource'}}, 'Environment': {'event_sinks': [], 'parameter_defaults': {}, 'parameters': {}, 'resource_registry': {'OS::Test::TestResource': 'https://server.test/nested.template', 'resources': {}}}}
    self.assertEqual(expected, res)