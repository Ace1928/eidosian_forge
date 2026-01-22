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
def test_resolve_attributes_duplicate_net_name(self):
    self.neutron_client.list_networks.return_value = {'networks': [{'id': 'fake_net_id', 'name': 'test'}, {'id': 'fake_net_id2', 'name': 'test'}]}
    self.fake_addresses = {'fake_net_id': [{'addr': '10.0.0.12'}], 'fake_net_id2': [{'addr': '10.100.0.12'}]}
    self.fake_extended_addresses = {'fake_net_id': [{'addr': '10.0.0.12'}], 'fake_net_id2': [{'addr': '10.100.0.12'}], 'test': [{'addr': '10.0.0.12'}, {'addr': '10.100.0.12'}]}
    c = self._create_resource('container', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(c.create)()
    self._mock_get_client()
    self._assert_addresses(self.fake_extended_addresses, c._resolve_attribute(container.Container.ADDRESSES))