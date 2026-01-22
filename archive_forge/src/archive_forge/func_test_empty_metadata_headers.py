import os
from glance.common import crypt
from glance.common import utils
from glance.tests import utils as test_utils
def test_empty_metadata_headers(self):
    """Ensure unset metadata is not encoded in HTTP headers"""
    metadata = {'foo': 'bar', 'snafu': None, 'bells': 'whistles', 'unset': None, 'empty': '', 'properties': {'distro': '', 'arch': None, 'user': 'nobody'}}
    headers = utils.image_meta_to_http_headers(metadata)
    self.assertNotIn('x-image-meta-snafu', headers)
    self.assertNotIn('x-image-meta-uset', headers)
    self.assertNotIn('x-image-meta-snafu', headers)
    self.assertNotIn('x-image-meta-property-arch', headers)
    self.assertEqual('bar', headers.get('x-image-meta-foo'))
    self.assertEqual('whistles', headers.get('x-image-meta-bells'))
    self.assertEqual('', headers.get('x-image-meta-empty'))
    self.assertEqual('', headers.get('x-image-meta-property-distro'))
    self.assertEqual('nobody', headers.get('x-image-meta-property-user'))