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
def test_endpoint(self):
    endp = endpoint.Endpoint(url='http://192.168.0.1', name='endpoint name', port='8080')
    self.assertEqual(True, endp.is_valid())
    dict_endp = endp.as_dict()
    for key in endpoint.ENDPOINT_KEYNAMES:
        self.assertIn(key, dict_endp)