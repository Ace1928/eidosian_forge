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
def test_server_update_None_networks_with_network_id(self):
    return_server = self.fc.servers.list()[3]
    return_server.id = '9102'
    self.patchobject(neutronclient.Client, 'create_port', return_value={'port': {'id': 'abcd1234'}})
    server = self._create_test_server(return_server, 'networks_update')
    new_networks = [{'network': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'fixed_ip': '1.2.3.4'}]
    update_props = self.server_props.copy()
    update_props['networks'] = new_networks
    update_template = server.t.freeze(properties=update_props)
    self.patchobject(self.fc.servers, 'get', return_value=return_server)
    iface = create_fake_iface(port='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', net='450abbc9-9b6d-4d6f-8c3a-c47ac34100ef', ip='1.2.3.4')
    self.patchobject(return_server, 'interface_list', return_value=[iface])
    mock_detach = self.patchobject(return_server, 'interface_detach')
    mock_attach = self.patchobject(return_server, 'interface_attach')
    mock_detach_check = self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
    mock_attach_check = self.patchobject(nova.NovaClientPlugin, 'check_interface_attach', return_value=True)
    scheduler.TaskRunner(server.update, update_template)()
    self.assertEqual((server.UPDATE, server.COMPLETE), server.state)
    self.assertEqual(1, mock_detach.call_count)
    self.assertEqual(1, mock_attach.call_count)
    self.assertEqual(1, mock_detach_check.call_count)
    self.assertEqual(1, mock_attach_check.call_count)