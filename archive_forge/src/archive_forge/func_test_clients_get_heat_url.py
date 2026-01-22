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
def test_clients_get_heat_url(self):
    con = mock.Mock()
    con.tenant_id = 'b363706f891f48019483f8bd6503c54b'
    c = clients.Clients(con)
    con.clients = c
    obj = c.client_plugin('heat')
    obj._get_client_option = mock.Mock()
    obj._get_client_option.return_value = None
    obj.url_for = mock.Mock(name='url_for')
    obj.url_for.return_value = 'url_from_keystone'
    self.assertEqual('url_from_keystone', obj.get_heat_url())
    heat_url = 'http://0.0.0.0:8004/v1/%(tenant_id)s'
    obj._get_client_option.return_value = heat_url
    tenant_id = 'b363706f891f48019483f8bd6503c54b'
    result = heat_url % {'tenant_id': tenant_id}
    self.assertEqual(result, obj.get_heat_url())
    obj._get_client_option.return_value = result
    self.assertEqual(result, obj.get_heat_url())