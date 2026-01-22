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
def test_server_update_server_flavor_failed(self):
    """Check raising exception due to resize call failing.

        If the status after a resize is not VERIFY_RESIZE, it means the resize
        call failed, so we raise an explicit error.
        """
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    server = self._create_test_server(return_server, 'srv_update2')
    update_props = self.server_props.copy()
    update_props['flavor'] = 'm1.small'
    update_template = server.t.freeze(properties=update_props)
    self.patchobject(self.fc.servers, 'get', side_effect=ServerStatus(return_server, ['RESIZE', 'ERROR']))
    mock_post = self.patchobject(self.fc.client, 'post_servers_1234_action', return_value=(202, None))
    updater = scheduler.TaskRunner(server.update, update_template)
    error = self.assertRaises(exception.ResourceFailure, updater)
    self.assertEqual("Error: resources.srv_update2: Resizing to '2' failed, status 'ERROR'", str(error))
    self.assertEqual((server.UPDATE, server.FAILED), server.state)
    mock_post.assert_called_once_with(body={'resize': {'flavorRef': '2'}})