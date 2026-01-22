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
def test_server_validate_connection_error_retry_successful(self):
    stack_name = 'srv_val'
    tmpl, stack = self._setup_test_stack(stack_name)
    tmpl.t['Resources']['WebServer']['Properties']['personality'] = {'/fake/path1': 'a' * 10}
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    self.patchobject(nova.NovaClientPlugin, 'is_version_supported', return_value=False)
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('server_create_image_err', resource_defns['WebServer'], stack)
    self.patchobject(self.fc.limits, 'get', side_effect=[requests.ConnectionError(), self.limits])
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=self.mock_image)
    self.assertIsNone(server.validate())