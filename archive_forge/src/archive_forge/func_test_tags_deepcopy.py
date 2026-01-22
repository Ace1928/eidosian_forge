import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_tags_deepcopy(self):
    proxy = glance.quota.QuotaImageTagsProxy(set(['a', 'b']))
    proxy_copy = copy.deepcopy(proxy)
    self.assertEqual(set(['a', 'b']), proxy_copy.tags)
    self.assertIn('a', proxy_copy)
    proxy_copy.remove('a')
    self.assertNotIn('a', proxy_copy)