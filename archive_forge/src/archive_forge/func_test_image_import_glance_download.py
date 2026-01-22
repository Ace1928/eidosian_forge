import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_image_import_glance_download(self):
    region = 'REGION2'
    remote_image_id = '75baf7b6-253a-11ed-8307-4b1057986a78'
    image_id = '606b0e88-7c5a-4d54-b5bb-046105d4de6f'
    service_interface = 'public'
    data = [('method', {'name': 'glance-download', 'glance_region': region, 'glance_image_id': remote_image_id, 'glance_service_interface': service_interface})]
    self.controller.image_import(image_id, 'glance-download', remote_region=region, remote_image_id=remote_image_id, remote_service_interface=service_interface)
    expect = [('POST', '/v2/images/%s/import' % image_id, {}, data)]
    self.assertEqual(expect, self.api.calls)