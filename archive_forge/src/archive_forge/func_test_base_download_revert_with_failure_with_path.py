import os
from unittest import mock
from glance_store import backend
from oslo_config import cfg
from taskflow.types import failure
from glance.async_.flows import api_image_import
import glance.common.exception
from glance import domain
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('glance.async_.flows._internal_plugins.base_download.store_api')
def test_base_download_revert_with_failure_with_path(self, mock_store_api):
    result = failure.Failure.from_exception(glance.common.exception.ImportTaskError())
    self.base_download_task.revert(result)
    mock_store_api.delete_from_backend.assert_called_once_with('/path/to_downloaded_data')