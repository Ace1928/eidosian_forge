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
def test_serialize_dict(self):
    original = collections.OrderedDict([('test_key', collections.OrderedDict([('a', 'b'), ('c', 'd')]))])
    expected = {'test_key': '{"a": "b", "c": "d"}'}
    actual = self.nova_plugin.meta_serialize(original)
    self.assertEqual(json.loads(expected['test_key']), json.loads(actual['test_key']))