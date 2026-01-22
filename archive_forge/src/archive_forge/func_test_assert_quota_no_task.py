import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_assert_quota_no_task(self):
    ignored = mock.MagicMock()
    task_repo = mock.MagicMock()
    task_repo.get.return_value = None
    task_id = 'some-task'
    enforce_fn = mock.MagicMock()
    enforce_fn.side_effect = exception.LimitExceeded
    with mock.patch.object(import_flow, 'LOG') as mock_log:
        self.assertRaises(exception.LimitExceeded, import_flow.assert_quota, ignored, task_repo, task_id, [], ignored, enforce_fn)
    task_repo.get.assert_called_once_with('some-task')
    mock_log.error.assert_called_once_with('Failed to find task %r to update after quota failure', 'some-task')
    task_repo.save.assert_not_called()