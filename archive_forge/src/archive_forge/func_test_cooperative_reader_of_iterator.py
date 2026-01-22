import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_cooperative_reader_of_iterator(self):
    """Ensure cooperative reader supports iterator backends too"""
    data = b'abcdefgh'
    data_list = [data[i:i + 1] * 3 for i in range(len(data))]
    reader = utils.CooperativeReader(data_list)
    chunks = []
    while True:
        chunks.append(reader.read(3))
        if chunks[-1] == b'':
            break
    meat = b''.join(chunks)
    self.assertEqual(b'aaabbbcccdddeeefffggghhh', meat)