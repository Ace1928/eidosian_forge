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
def test_ct_create(self):
    self._create_ct(self.t)
    args = {'name': 'test-cluster-template', 'plugin_name': 'vanilla', 'hadoop_version': '2.3.0', 'description': '', 'default_image_id': None, 'net_id': 'some_network_id', 'anti_affinity': None, 'node_groups': None, 'cluster_configs': None, 'use_autoconfig': None, 'shares': [{'id': 'e45eaabf-9300-42e2-b6eb-9ebc92081f46', 'access_level': 'ro', 'path': None}]}
    self.ct_mgr.create.assert_called_once_with(**args)