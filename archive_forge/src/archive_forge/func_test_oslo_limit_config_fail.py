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
@mock.patch('oslo_limit.limit.Enforcer')
@mock.patch.object(ks_quota, 'LOG')
def test_oslo_limit_config_fail(self, mock_LOG, mock_enforcer):
    self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
    self.config(use_keystone_limits=True)
    mock_enforcer.return_value.enforce.side_effect = ol_exc.SessionInitError('test')
    context = FakeContext()
    self._create_fake_image(context, 100)
    self.assertRaises(ol_exc.SessionInitError, ks_quota.enforce_image_size_total, context, context.owner)
    mock_LOG.error.assert_called_once_with('Failed to initialize oslo_limit, likely due to incorrect or insufficient configuration: %(err)s', {'err': "Can't initialise OpenStackSDK session: test."})