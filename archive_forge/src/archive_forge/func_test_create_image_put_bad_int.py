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
def test_create_image_put_bad_int(self):
    self.cloud.image_api_use_tasks = False
    self.assertRaises(exceptions.SDKException, self._call_create_image, self.image_name, allow_duplicates=True, min_disk='fish', min_ram=0)
    self.assert_calls()