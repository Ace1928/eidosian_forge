import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_custom_register(self):
    registry = msgpackutils.default_registry.copy(unfreeze=True)
    registry.register(ColorHandler())
    c = Color(255, 254, 253)
    c_b = msgpackutils.dumps(c, registry=registry)
    c = msgpackutils.loads(c_b, registry=registry)
    self.assertEqual(255, c.r)
    self.assertEqual(254, c.g)
    self.assertEqual(253, c.b)