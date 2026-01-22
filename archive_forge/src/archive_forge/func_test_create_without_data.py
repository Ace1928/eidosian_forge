import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_create_without_data(self):
    params = {'id': '1', 'name': 'image-1', 'container_format': 'ovf', 'disk_format': 'vhd', 'owner': 'asdf', 'size': 1024, 'min_ram': 512, 'min_disk': 10, 'copy_from': 'http://example.com', 'properties': {'a': 'b', 'c': 'd'}}
    image = self.mgr.create(**params)
    expect_headers = {'x-image-meta-id': '1', 'x-image-meta-name': 'image-1', 'x-image-meta-container_format': 'ovf', 'x-image-meta-disk_format': 'vhd', 'x-image-meta-owner': 'asdf', 'x-image-meta-size': '1024', 'x-image-meta-min_ram': '512', 'x-image-meta-min_disk': '10', 'x-glance-api-copy-from': 'http://example.com', 'x-image-meta-property-a': 'b', 'x-image-meta-property-c': 'd'}
    expect = [('POST', '/v1/images', expect_headers, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual('1', image.id)
    self.assertEqual('image-1', image.name)
    self.assertEqual('ovf', image.container_format)
    self.assertEqual('vhd', image.disk_format)
    self.assertEqual('asdf', image.owner)
    self.assertEqual(1024, image.size)
    self.assertEqual(512, image.min_ram)
    self.assertEqual(10, image.min_disk)
    self.assertEqual(False, image.is_public)
    self.assertEqual(False, image.protected)
    self.assertEqual(False, image.deleted)
    self.assertEqual({'a': 'b', 'c': 'd'}, image.properties)