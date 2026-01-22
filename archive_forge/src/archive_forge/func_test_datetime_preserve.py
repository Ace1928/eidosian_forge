import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_datetime_preserve(self):
    x = datetime.datetime(1920, 2, 3, 4, 5, 6, 7)
    self.assertEqual(x, _dumps_loads(x))