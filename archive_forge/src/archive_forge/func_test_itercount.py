import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_itercount(self):
    it = itertools.count(1)
    next(it)
    next(it)
    it2 = _dumps_loads(it)
    self.assertEqual(next(it), next(it2))
    it = itertools.count(0)
    it2 = _dumps_loads(it)
    self.assertEqual(next(it), next(it2))