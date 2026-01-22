from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_validate_vlan_type(self):
    self.t = template_format.parse(inline_templates.SEGMENT_TEMPLATE)
    props = self.t['resources']['segment']['properties']
    props['network_type'] = 'vlan'
    self.stack = utils.parse_stack(self.t)
    rsrc = self.stack['segment']
    errMsg = 'physical_network is required for vlan provider network.'
    error = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertEqual(errMsg, str(error))
    props['physical_network'] = 'physnet'
    props['segmentation_id'] = '4095'
    self.stack = utils.parse_stack(self.t)
    rsrc = self.stack['segment']
    errMsg = 'Up to 4094 VLAN network segments can exist on each physical_network.'
    error = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertEqual(errMsg, str(error))