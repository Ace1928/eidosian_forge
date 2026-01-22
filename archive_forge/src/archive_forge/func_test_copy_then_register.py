import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_copy_then_register(self):
    registry = msgpackutils.default_registry
    self.assertRaises(ValueError, registry.register, MySpecialSetHandler(), reserved=True, override=True)
    registry = registry.copy(unfreeze=True)
    registry.register(MySpecialSetHandler(), reserved=True, override=True)
    h = registry.match(set())
    self.assertIsInstance(h, MySpecialSetHandler)