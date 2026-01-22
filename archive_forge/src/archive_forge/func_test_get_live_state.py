from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_get_live_state(self):
    ngt = self._create_ngt(self.t)
    resp = mock.MagicMock()
    resp.to_dict.return_value = {'volume_local_to_instance': False, 'availability_zone': None, 'updated_at': None, 'use_autoconfig': True, 'volumes_per_node': 0, 'id': '6157755e-dfd3-45b4-a445-36588e5f75ad', 'security_groups': None, 'shares': None, 'node_configs': {}, 'auto_security_group': False, 'volumes_availability_zone': None, 'description': '', 'volume_mount_prefix': '/volumes/disk', 'plugin_name': 'vanilla', 'floating_ip_pool': None, 'is_default': False, 'image_id': None, 'volumes_size': 0, 'is_proxy_gateway': False, 'is_public': False, 'hadoop_version': '2.7.1', 'name': 'cluster-nodetemplate-jlgzovdaivn', 'tenant_id': '221b4f51e9bd4f659845f657a3051a46', 'created_at': '2016-01-29T11:08:46', 'volume_type': None, 'is_protected': False, 'node_processes': ['namenode'], 'flavor_id': '2'}
    self.ngt_mgr.get.return_value = resp
    ngt.properties.data['flavor'] = '1'
    reality = ngt.get_live_state(ngt.properties)
    expected = {'volume_local_to_instance': False, 'availability_zone': None, 'use_autoconfig': True, 'volumes_per_node': 0, 'security_groups': None, 'shares': None, 'node_configs': {}, 'auto_security_group': False, 'volumes_availability_zone': None, 'description': '', 'plugin_name': 'vanilla', 'floating_ip_pool': None, 'image_id': None, 'volumes_size': 0, 'is_proxy_gateway': False, 'hadoop_version': '2.7.1', 'name': 'cluster-nodetemplate-jlgzovdaivn', 'volume_type': None, 'node_processes': ['namenode'], 'flavor': '2'}
    self.assertEqual(expected, reality)
    ngt.properties.data['flavor'] = '2'
    reality = ngt.get_live_state(ngt.properties)
    self.assertEqual('2', reality.get('flavor'))