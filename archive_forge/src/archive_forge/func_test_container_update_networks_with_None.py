import copy
from unittest import mock
from oslo_config import cfg
from zunclient import exceptions as zc_exc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import zun
from heat.engine.resources.openstack.zun import container
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_container_update_networks_with_None(self):
    new_networks = None
    self._test_container_update_networks(new_networks)
    self.assertEqual(3, self.mock_detach.call_count)
    self.assertEqual(1, self.mock_attach.call_count)
    self.assertEqual(3, self.mock_detach_check.call_count)
    self.assertEqual(1, self.mock_attach_check.call_count)