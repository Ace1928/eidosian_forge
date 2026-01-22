from unittest import mock
from aodhclient import exceptions as aodh_exc
from cinderclient import exceptions as cinder_exc
from glanceclient import exc as glance_exc
from heatclient import client as heatclient
from heatclient import exc as heat_exc
from keystoneauth1 import exceptions as keystone_exc
from keystoneauth1.identity import generic
from manilaclient import exceptions as manila_exc
from mistralclient.api import base as mistral_base
from neutronclient.common import exceptions as neutron_exc
from openstack import exceptions
from oslo_config import cfg
from saharaclient.api import base as sahara_base
from swiftclient import exceptions as swift_exc
from testtools import testcase
from troveclient import client as troveclient
from zaqarclient.transport import errors as zaqar_exc
from heat.common import exception
from heat.engine import clients
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.tests import common
from heat.tests import fakes
from heat.tests.openstack.nova import fakes as fakes_nova
@testcase.skip('skipped until keystone can read context auth_ref')
def test_create_all_clients(self):
    con = mock.Mock()
    con.auth_url = 'http://auth.example.com:5000/v2.0'
    con.tenant_id = 'b363706f891f48019483f8bd6503c54b'
    con.auth_token = '3bcc3d3a03f44e3d8377f9247b0ad155'
    c = clients.Clients(con)
    con.clients = c
    for plugin_name in clients._mgr.names():
        self.assertTrue(clients.has_client(plugin_name))
        c.client(plugin_name)