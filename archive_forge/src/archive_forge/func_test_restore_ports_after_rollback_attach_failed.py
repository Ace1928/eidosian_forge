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
@mock.patch.object(server_network_mixin.ServerNetworkMixin, 'store_external_ports')
def test_restore_ports_after_rollback_attach_failed(self, store_ports):
    t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
    server.resource_id = 'existing_server'
    port_ids = [{'id': 1122}, {'id': 3344}]
    server._data = {'internal_ports': jsonutils.dumps(port_ids)}
    self.patchobject(nova.NovaClientPlugin, '_check_active')
    nova.NovaClientPlugin._check_active.return_value = True
    old_server = mock.Mock()
    old_server.resource_id = 'old_server'
    stack._backup_stack = mock.Mock()
    stack._backup_stack().resources.get.return_value = old_server
    old_server._data_get_ports.side_effect = [port_ids, []]

    class Fake(object):

        def interface_list(self):
            return [iface(1122)]
    iface = collections.namedtuple('iface', ['port_id'])
    self.patchobject(nova.NovaClientPlugin, 'interface_detach')
    self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
    self.patchobject(nova.NovaClientPlugin, 'interface_attach')
    self.patchobject(nova.NovaClientPlugin, 'fetch_server')
    self.patchobject(nova.NovaClientPlugin.check_interface_attach.retry, 'sleep')
    nova.NovaClientPlugin.fetch_server.side_effect = [Fake()] * 11
    exc = self.assertRaises(exception.InterfaceAttachFailed, server.restore_prev_rsrc)
    self.assertIn('Failed to attach interface (3344) to server (old_server)', str(exc))