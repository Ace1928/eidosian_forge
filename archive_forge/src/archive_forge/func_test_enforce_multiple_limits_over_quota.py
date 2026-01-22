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
def test_enforce_multiple_limits_over_quota(self):
    self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
    self.config(use_keystone_limits=True)
    context = FakeContext()
    self.assertRaises(exception.LimitExceeded, ks_quota._enforce_some, context, context.owner, {ks_quota.QUOTA_IMAGE_SIZE_TOTAL: lambda: 200, 'another_limit': lambda: 1}, {'another_limit': 5})