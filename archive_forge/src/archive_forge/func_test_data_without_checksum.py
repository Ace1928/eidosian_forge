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
def test_data_without_checksum(self):
    body = self.controller.data('5cc4bebc-db27-11e1-a1eb-080027cbe205', do_checksum=False)
    body = ''.join([b for b in body])
    self.assertEqual('A', body)
    body = self.controller.data('5cc4bebc-db27-11e1-a1eb-080027cbe205')
    body = ''.join([b for b in body])
    self.assertEqual('A', body)