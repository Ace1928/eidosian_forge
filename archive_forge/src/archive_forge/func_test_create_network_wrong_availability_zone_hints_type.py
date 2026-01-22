import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_wrong_availability_zone_hints_type(self):
    azh_opts = 'invalid'
    with testtools.ExpectedException(exceptions.SDKException, "Parameter 'availability_zone_hints' must be a list"):
        self.cloud.create_network('netname', availability_zone_hints=azh_opts)