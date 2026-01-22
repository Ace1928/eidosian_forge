import io
import operator
import tempfile
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import _log
from openstack import exceptions
from openstack.image.v2 import image
from openstack.tests.unit import base
from openstack import utils
def test_download_checksum_match(self):
    sot = image.Image(**EXAMPLE)
    resp = FakeResponse(b'abc', headers={'Content-MD5': '900150983cd24fb0d6963f7d28e17f72', 'Content-Type': 'application/octet-stream'})
    self.sess.get.return_value = resp
    rv = sot.download(self.sess)
    self.sess.get.assert_called_with('images/IDENTIFIER/file', stream=False)
    self.assertEqual(rv, resp)