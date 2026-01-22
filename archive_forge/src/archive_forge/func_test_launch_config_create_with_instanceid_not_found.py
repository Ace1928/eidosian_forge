from unittest import mock
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_launch_config_create_with_instanceid_not_found(self):
    t = template_format.parse(inline_templates.as_template)
    lcp = t['Resources']['LaunchConfig']['Properties']
    lcp['InstanceId'] = '5678'
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    rsrc = stack['LaunchConfig']
    self.stub_SnapshotConstraint_validate()
    self.stub_ImageConstraint_validate()
    self.stub_FlavorConstraint_validate()
    self.patchobject(nova.NovaClientPlugin, 'get_server', side_effect=exception.EntityNotFound(entity='Server', name='5678'))
    msg = "Property error: Resources.LaunchConfig.Properties.InstanceId: Error validating value '5678': The Server (5678) could not be found."
    exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertIn(msg, str(exc))