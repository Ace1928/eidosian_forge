import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_scaling_group_create_with_instanceid_not_found(self):
    t = template_format.parse(as_template)
    agp = t['Resources']['WebServerGroup']['Properties']
    agp.pop('LaunchConfigurationName')
    agp['InstanceId'] = '5678'
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    rsrc = stack['WebServerGroup']
    self._stub_nova_server_get(not_found=True)
    msg = "Property error: Resources.WebServerGroup.Properties.InstanceId: Error validating value '5678': The Server (5678) could not be found."
    exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertIn(msg, str(exc))