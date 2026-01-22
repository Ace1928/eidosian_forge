import time
from unittest import mock
import uuid
from pycadf import attachment
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import event
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
from pycadf import measurement
from pycadf import metric
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf.tests import base
from pycadf import timestamp
def test_host(self):
    h = host.Host(id=identifier.generate_uuid(), address='192.168.0.1', agent='client', platform='AIX')
    self.assertEqual(True, h.is_valid())
    dict_host = h.as_dict()
    for key in host.HOST_KEYNAMES:
        self.assertIn(key, dict_host)