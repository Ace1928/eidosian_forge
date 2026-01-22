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
def test_segment_handle_create(self):
    seg = mock.Mock(id='9c1eb3fe-7bba-479d-bd43-1d497e53c384')
    create_props = {'name': 'test_segment', 'network_id': 'private', 'network_type': 'vxlan', 'segmentation_id': 101}
    mock_create = self.patchobject(self.sdkclient.network, 'create_segment', return_value=seg)
    self.segment.handle_create()
    self.assertEqual('9c1eb3fe-7bba-479d-bd43-1d497e53c384', self.segment.resource_id)
    mock_create.assert_called_once_with(**create_props)