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
def test_validate_flat_type(self):
    self.t = template_format.parse(inline_templates.SEGMENT_TEMPLATE)
    props = self.t['resources']['segment']['properties']
    props['network_type'] = 'flat'
    props['physical_network'] = 'physnet'
    self.stack = utils.parse_stack(self.t)
    rsrc = self.stack['segment']
    errMsg = 'segmentation_id is prohibited for flat provider network.'
    error = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertEqual(errMsg, str(error))