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
def test_data_with_wrong_checksum(self):
    body = self.controller.data('headeronly-db27-11e1-a1eb-080027cbe205', do_checksum=False)
    body = ''.join([b for b in body])
    self.assertEqual('BB', body)
    body = self.controller.data('headeronly-db27-11e1-a1eb-080027cbe205')
    try:
        body = ''.join([b for b in body])
        self.fail('data did not raise an error.')
    except IOError as e:
        self.assertEqual(errno.EPIPE, e.errno)
        msg = 'was 9d3d9048db16a7eee539e93e3618cbe7 expected wrong'
        self.assertIn(msg, str(e))
    body = self.controller.data('chkonly-db27-11e1-a1eb-080027cbe205', do_checksum=False)
    body = ''.join([b for b in body])
    self.assertEqual('BB', body)
    body = self.controller.data('chkonly-db27-11e1-a1eb-080027cbe205')
    try:
        body = ''.join([b for b in body])
        self.fail('data did not raise an error.')
    except IOError as e:
        self.assertEqual(errno.EPIPE, e.errno)
        msg = 'was 9d3d9048db16a7eee539e93e3618cbe7 expected wrong'
        self.assertIn(msg, str(e))
    body = self.controller.data('multihash-db27-11e1-a1eb-080027cbe205', do_checksum=False)
    body = ''.join([b for b in body])
    self.assertEqual('BB', body)
    body = self.controller.data('multihash-db27-11e1-a1eb-080027cbe205')
    try:
        body = ''.join([b for b in body])
        self.fail('data did not raise an error.')
    except IOError as e:
        self.assertEqual(errno.EPIPE, e.errno)
        msg = 'was 9d3d9048db16a7eee539e93e3618cbe7 expected junk'
        self.assertIn(msg, str(e))
    body = self.controller.data('badalgo-db27-11e1-a1eb-080027cbe205', do_checksum=False)
    body = ''.join([b for b in body])
    self.assertEqual('BB', body)
    try:
        body = self.controller.data('badalgo-db27-11e1-a1eb-080027cbe205')
        self.fail('bad os_hash_algo did not raise an error.')
    except ValueError as e:
        msg = 'unsupported hash type not_an_algo'
        self.assertIn(msg, str(e))