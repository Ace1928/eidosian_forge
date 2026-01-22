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
@mock.patch.object(nova.NovaClientPlugin, 'client')
def test_validate_insufficient_disk_flavor(self, mock_create):
    stack_name = 'test_stack'
    tmpl, stack = self._setup_test_stack(stack_name)
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('server_insufficient_disk_flavor', resource_defns['WebServer'], stack)
    mock_image = mock.Mock(min_ram=1, status='active', min_disk=100)
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=mock_image)
    self.patchobject(nova.NovaClientPlugin, 'get_flavor', return_value=self.mock_flavor)
    error = self.assertRaises(exception.StackValidationFailed, server.validate)
    self.assertEqual('Image F18-x86_64-gold requires 100 GB minimum disk space. Flavor m1.large has only 4 GB.', str(error))