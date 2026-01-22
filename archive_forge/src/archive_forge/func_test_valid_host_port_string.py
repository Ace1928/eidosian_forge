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
def test_valid_host_port_string(self):
    valid_pairs = ['10.11.12.13:80', '172.17.17.1:65535', '[fe80::a:b:c:d]:9990', 'localhost:9990', 'localhost.localdomain:9990', 'glance02.stack42.local:1234', 'glance04-a.stack47.local:1234', 'img83.glance.xn--penstack-r74e.org:13080']
    for pair_str in valid_pairs:
        host, port = utils.parse_valid_host_port(pair_str)
        escaped = pair_str.startswith('[')
        expected_host = '%s%s%s' % ('[' if escaped else '', host, ']' if escaped else '')
        self.assertTrue(pair_str.startswith(expected_host))
        self.assertGreater(port, 0)
        expected_pair = '%s:%d' % (expected_host, port)
        self.assertEqual(expected_pair, pair_str)