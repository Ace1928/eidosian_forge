import io
import operator
import tempfile
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack import connection
from openstack import exceptions
from openstack.image.v1 import image as image_v1
from openstack.image.v2 import image
from openstack.tests import fakes
from openstack.tests.unit import base
def test_download_image_two_outputs(self):
    fake_fd = io.BytesIO()
    self.assertRaises(exceptions.SDKException, self.cloud.download_image, self.image_name, output_path='fake_path', output_file=fake_fd)