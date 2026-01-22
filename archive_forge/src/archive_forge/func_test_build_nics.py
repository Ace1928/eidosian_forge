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
def test_build_nics(self):
    return_server = self.fc.servers.list()[1]
    server = self._create_test_server(return_server, 'test_server_create')
    self.patchobject(neutronclient.Client, 'create_port', return_value={'port': {'id': '4815162342'}})
    self.assertIsNone(server._build_nics([]))
    self.assertIsNone(server._build_nics(None))
    self.assertEqual([{'port-id': 'aaaabbbb', 'net-id': None, 'tag': 'nic1'}, {'v4-fixed-ip': '192.0.2.0', 'net-id': None}], server._build_nics([{'port': 'aaaabbbb', 'tag': 'nic1'}, {'fixed_ip': '192.0.2.0'}]))
    self.assertEqual([{'port-id': 'aaaabbbb', 'net-id': None}, {'port-id': 'aaaabbbb', 'net-id': None}], server._build_nics([{'port': 'aaaabbbb', 'fixed_ip': '192.0.2.0'}, {'port': 'aaaabbbb', 'fixed_ip': '2002::2'}]))
    self.assertEqual([{'port-id': 'aaaabbbb', 'net-id': None}, {'v6-fixed-ip': '2002::2', 'net-id': None}], server._build_nics([{'port': 'aaaabbbb'}, {'fixed_ip': '2002::2'}]))
    self.assertEqual([{'net-id': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}], server._build_nics([{'network': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}]))