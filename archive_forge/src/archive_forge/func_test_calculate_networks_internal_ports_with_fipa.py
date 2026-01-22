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
def test_calculate_networks_internal_ports_with_fipa(self):
    tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n                  floating_ip: 1199\n                - network: 8765\n                  subnet: 5678\n                  fixed_ip: 127.0.0.2\n                  floating_ip: 9911\n        '
    t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
    self.patchobject(server, 'update_networks_matching_iface_port')
    server._data = {'internal_ports': '[{"id": "1122"}]'}
    self.port_create.return_value = {'port': {'id': '5566'}}
    self.patchobject(resource.Resource, 'data_set')
    self.resolve.side_effect = ['0912', '9021']
    fipa = self.patchobject(neutronclient.Client, 'update_floatingip', side_effect=[neutronclient.exceptions.NotFound, '9911', '11910', '1199'])
    old_net = [self.create_old_net(net='4321', subnet='1234', ip='127.0.0.1', port='1122', floating_ip='1199'), self.create_old_net(net='8765', subnet='5678', ip='127.0.0.2', port='3344', floating_ip='9911')]
    interfaces = [create_fake_iface(port='1122', net='4321', ip='127.0.0.1', subnet='1234'), create_fake_iface(port='3344', net='8765', ip='127.0.0.2', subnet='5678')]
    new_net = [{'network': '8765', 'subnet': '5678', 'fixed_ip': '127.0.0.2', 'port': '3344', 'floating_ip': '11910'}, {'network': '0912', 'subnet': '9021', 'fixed_ip': '127.0.0.1', 'floating_ip': '1199', 'port': '1122'}]
    server.calculate_networks(old_net, new_net, interfaces)
    fipa.assert_has_calls((mock.call('1199', {'floatingip': {'port_id': None}}), mock.call('9911', {'floatingip': {'port_id': None}}), mock.call('11910', {'floatingip': {'port_id': '3344', 'fixed_ip_address': '127.0.0.2'}}), mock.call('1199', {'floatingip': {'port_id': '1122', 'fixed_ip_address': '127.0.0.1'}})))