from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def test_with_admin_repo(self):
    admin_repo = mock.MagicMock()
    executor = glance.async_.TaskExecutor(self.context, self.task_repo, self.image_repo, self.image_factory, admin_repo=admin_repo)
    self.assertEqual(admin_repo, executor.admin_repo)