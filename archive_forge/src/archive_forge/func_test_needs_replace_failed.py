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
def test_needs_replace_failed(self):
    self.stack.store()
    self.segment.state_set(self.segment.CREATE, self.segment.FAILED)
    side_effect = [exceptions.ResourceNotFound, 'attr']
    mock_show_resource = self.patchobject(self.segment, '_show_resource', side_effect=side_effect)
    self.segment.resource_id = None
    self.assertTrue(self.segment.needs_replace_failed())
    self.assertEqual(0, mock_show_resource.call_count)
    self.segment.resource_id = 'seg_id'
    self.assertTrue(self.segment.needs_replace_failed())
    self.assertEqual(1, mock_show_resource.call_count)
    self.assertFalse(self.segment.needs_replace_failed())
    self.assertEqual(2, mock_show_resource.call_count)