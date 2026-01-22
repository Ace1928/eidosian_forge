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
@ddt.data('headeronly', 'chkonly', 'multihash')
def test_data_with_checksum_but_no_md5_algo(self, prefix):
    with mock.patch('hashlib.new', mock.MagicMock(side_effect=ValueError('unsupported hash type'))):
        body = self.controller.data(prefix + '-dd57-11e1-af0f-02163e68b1d8', allow_md5_fallback=True)
        try:
            body = ''.join([b for b in body])
            self.fail('missing md5 algo did not raise an error')
        except IOError as e:
            self.assertEqual(errno.EPIPE, e.errno)
            msg = 'md5 algorithm is not available on the client'
            self.assertIn(msg, str(e))