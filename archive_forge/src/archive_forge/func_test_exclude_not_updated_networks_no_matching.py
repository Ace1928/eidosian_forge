import collections
import contextlib
import copy
from unittest import mock
from keystoneauth1 import exceptions as ks_exceptions
from neutronclient.v2_0 import client as neutronclient
from novaclient import exceptions as nova_exceptions
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import requests
from urllib import parse as urlparse
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.nova import server as servers
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_exclude_not_updated_networks_no_matching(self):
    return_server = self.fc.servers.list()[3]
    server = self._create_test_server(return_server, 'networks_update')
    for new_nets in ([], [{'port': '952fd4ae-53b9-4b39-9e5f-8929c553b5ae', 'network': '450abbc9-9b6d-4d6f-8c3a-c47ac34100dd'}]):
        old_nets = [self.create_old_net(port='2a60cbaa-3d33-4af6-a9ce-83594ac546fc'), self.create_old_net(net='f3ef5d2f-d7ba-4b27-af66-58ca0b81e032', ip='1.2.3.4'), self.create_old_net(net='0da8adbf-a7e2-4c59-a511-96b03d2da0d7')]
        interfaces = [create_fake_iface(port='2a60cbaa-3d33-4af6-a9ce-83594ac546fc', net='450abbc9-9b6d-4d6f-8c3a-c47ac34100aa', ip='4.3.2.1', subnet='subnetsu-bnet-subn-etsu-bnetsubnetsu'), create_fake_iface(port='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', net='f3ef5d2f-d7ba-4b27-af66-58ca0b81e032', ip='1.2.3.4', subnet='subnetsu-bnet-subn-etsu-bnetsubnetsu'), create_fake_iface(port='bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', net='0da8adbf-a7e2-4c59-a511-96b03d2da0d7', ip='4.2.3.1', subnet='subnetsu-bnet-subn-etsu-bnetsubnetsu')]
        new_nets_cpy = copy.deepcopy(new_nets)
        old_nets_cpy = copy.deepcopy(old_nets)
        old_nets_cpy[0]['fixed_ip'] = '4.3.2.1'
        old_nets_cpy[0]['network'] = '450abbc9-9b6d-4d6f-8c3a-c47ac34100aa'
        old_nets_cpy[0]['subnet'] = 'subnetsu-bnet-subn-etsu-bnetsubnetsu'
        old_nets_cpy[1]['fixed_ip'] = '1.2.3.4'
        old_nets_cpy[1]['port'] = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
        old_nets_cpy[1]['subnet'] = 'subnetsu-bnet-subn-etsu-bnetsubnetsu'
        old_nets_cpy[2]['fixed_ip'] = '4.2.3.1'
        old_nets_cpy[2]['port'] = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
        old_nets_cpy[2]['subnet'] = 'subnetsu-bnet-subn-etsu-bnetsubnetsu'
        for net in new_nets_cpy:
            for key in ('port', 'network', 'fixed_ip', 'uuid', 'subnet', 'port_extra_properties', 'floating_ip', 'allocate_network', 'tag'):
                net.setdefault(key)
        server._exclude_not_updated_networks(old_nets, new_nets, interfaces)
        self.assertEqual(old_nets_cpy, old_nets)
        self.assertEqual(new_nets_cpy, new_nets)