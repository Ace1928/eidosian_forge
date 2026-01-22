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
def test_launch_config_create_with_instanceid(self):
    t = template_format.parse(inline_templates.as_template)
    lcp = t['Resources']['LaunchConfig']['Properties']
    lcp['InstanceId'] = '5678'
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    rsrc = stack['LaunchConfig']
    lc_props = {'ImageId': 'foo', 'InstanceType': 'bar', 'BlockDeviceMappings': lcp['BlockDeviceMappings'], 'KeyName': 'hth_keypair', 'SecurityGroups': ['hth_test']}
    rsrc.rebuild_lc_properties = mock.Mock(return_value=lc_props)
    self.stub_ImageConstraint_validate()
    self.stub_FlavorConstraint_validate()
    self.stub_SnapshotConstraint_validate()
    self.stub_ServerConstraint_validate()
    self.assertIsNone(rsrc.validate())
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)