import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_update_with_data(self):
    image_data = io.StringIO('XXX')
    self.mgr.update('1', data=image_data)
    expect_headers = {'x-image-meta-size': '3', 'x-glance-registry-purge-props': 'false'}
    expect = [('PUT', '/v1/images/1', expect_headers, image_data)]
    self.assertEqual(expect, self.api.calls)