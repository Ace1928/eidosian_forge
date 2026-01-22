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
@mock.patch.object(servers.Server, 'prepare_for_replace')
def test_server_update_image_replace(self, mock_replace):
    stack_name = 'update_imgrep'
    tmpl, stack = self._setup_test_stack(stack_name)
    tmpl.t['Resources']['WebServer']['Properties']['image_update_policy'] = 'REPLACE'
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('server_update_image_replace', resource_defns['WebServer'], stack)
    image_id = self.getUniqueString()
    update_props = tmpl.t['Resources']['WebServer']['Properties'].copy()
    update_props['image'] = image_id
    update_template = server.t.freeze(properties=update_props)
    updater = scheduler.TaskRunner(server.update, update_template)
    self.assertRaises(resource.UpdateReplace, updater)