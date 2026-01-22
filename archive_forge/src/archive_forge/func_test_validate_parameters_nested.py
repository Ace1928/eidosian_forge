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
def test_validate_parameters_nested(self):
    t = template_format.parse(test_template_allowed_integers)
    other_template = test_template_no_default.replace('net_name', 'net_name2')
    files = {'env1': 'parameter_defaults:\n  net_name: net1', 'env2': 'parameter_defaults:\n  net_name: net2\n  net_name2: net3', 'tmpl1.yaml': test_template_no_default, 'tmpl2.yaml': other_template}
    params = {'parameters': {}, 'parameter_defaults': {}}
    ret = self.engine.validate_template(self.ctx, t, params=params, files=files, environment_files=['env1', 'env2'])
    self.assertEqual('net2', params['parameter_defaults']['net_name'])
    self.assertEqual('net3', params['parameter_defaults']['net_name2'])
    expected = {'Description': 'No description', 'Parameters': {'size': {'AllowedValues': [1, 4, 8], 'Description': '', 'Label': u'size', 'NoEcho': 'false', 'Type': 'Number'}}, 'Environment': {'event_sinks': [], 'parameter_defaults': {'net_name': u'net2', 'net_name2': u'net3'}, 'parameters': {}, 'resource_registry': {'resources': {}}}}
    self.assertEqual(expected, ret)