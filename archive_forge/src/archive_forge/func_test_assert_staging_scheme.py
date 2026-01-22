import os
from unittest import mock
import glance_store
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context
from glance import housekeeping
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('glance.common.store_utils.get_dir_separator')
def test_assert_staging_scheme(self, mock_get_dir_separator):
    mock_get_dir_separator.return_value = ('/', 'http://foo')
    self.assertRaises(exception.GlanceException, lambda: housekeeping.staging_store_path())