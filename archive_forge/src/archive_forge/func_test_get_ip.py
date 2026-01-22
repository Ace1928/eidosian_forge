import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_get_ip(self):
    my_image = mock.MagicMock()
    my_image.addresses = {'public': [{'version': 4, 'addr': '4.5.6.7'}, {'version': 6, 'addr': '2401:1801:7800:0101:c058:dd33:ff18:04e6'}], 'private': [{'version': 4, 'addr': '10.13.12.13'}]}
    expected = '4.5.6.7'
    observed = self.nova_plugin.get_ip(my_image, 'public', 4)
    self.assertEqual(expected, observed)
    expected = '10.13.12.13'
    observed = self.nova_plugin.get_ip(my_image, 'private', 4)
    self.assertEqual(expected, observed)
    expected = '2401:1801:7800:0101:c058:dd33:ff18:04e6'
    observed = self.nova_plugin.get_ip(my_image, 'public', 6)
    self.assertEqual(expected, observed)