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
def test_deactivate(self):
    sot = image.Image(**EXAMPLE)
    self.assertIsNone(sot.deactivate(self.sess))
    self.sess.post.assert_called_with('images/IDENTIFIER/actions/deactivate')