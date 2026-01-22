import binascii
import re
from six import StringIO
from boto import storage_uri
from boto.exception import BotoClientError
from boto.gs.acl import SupportedPermissions as perms
from tests.integration.gs.testcase import GSTestCase
def testCompose(self):
    data1 = 'hello '
    data2 = 'world!'
    expected_crc = 1238062967
    b = self._MakeBucket()
    bucket_uri = storage_uri('gs://%s' % b.name)
    key_uri1 = bucket_uri.clone_replace_name('component1')
    key_uri1.set_contents_from_string(data1)
    key_uri2 = bucket_uri.clone_replace_name('component2')
    key_uri2.set_contents_from_string(data2)
    key_uri_composite = bucket_uri.clone_replace_name('composite')
    components = [key_uri1, key_uri2]
    key_uri_composite.compose(components, content_type='text/plain')
    self.assertEquals(key_uri_composite.get_contents_as_string(), data1 + data2)
    composite_key = key_uri_composite.get_key()
    cloud_crc32c = binascii.hexlify(composite_key.cloud_hashes['crc32c'])
    self.assertEquals(cloud_crc32c, hex(expected_crc)[2:])
    self.assertEquals(composite_key.content_type, 'text/plain')
    key_uri1.bucket_name += '2'
    try:
        key_uri_composite.compose(components)
        self.fail("Composing between buckets didn't fail as expected.")
    except BotoClientError as err:
        self.assertEquals(err.reason, 'GCS does not support inter-bucket composing')