import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_provider_wrong_type(self):
    provider_opts = 'invalid'
    with testtools.ExpectedException(exceptions.SDKException, "Parameter 'provider' must be a dict"):
        self.cloud.create_network('netname', provider=provider_opts)