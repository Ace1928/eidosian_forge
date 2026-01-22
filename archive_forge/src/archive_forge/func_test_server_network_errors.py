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
def test_server_network_errors(self):
    stack_name = 'net_err'
    tmpl, stack = self._setup_test_stack(stack_name, test_templ=ns_template)
    resolver = self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id')
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('server', resource_defns['server'], stack)
    resolver.side_effect = neutron.exceptions.NotFound()
    server.reparse()
    self.assertRaises(ValueError, server.properties.get, 'networks')
    resolver.side_effect = neutron.exceptions.NeutronClientNoUniqueMatch()
    ex = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(server.create))
    self.assertIn('use an ID to be more specific.', str(ex))