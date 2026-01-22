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
def test_empty_instance_user(self):
    """Test Nova server doesn't set instance_user in build_userdata

        Launching the instance should not pass any user name to
        build_userdata. The default cloud-init user set up for the image
        will be used instead.
        """
    return_server = self.fc.servers.list()[1]
    server = self._setup_test_server(return_server, 'without_user')
    metadata = server.metadata_get()
    build_data = self.patchobject(nova.NovaClientPlugin, 'build_userdata')
    scheduler.TaskRunner(server.create)()
    build_data.assert_called_with(metadata, 'wordpress', instance_user=None, user_data_format='HEAT_CFNTOOLS')