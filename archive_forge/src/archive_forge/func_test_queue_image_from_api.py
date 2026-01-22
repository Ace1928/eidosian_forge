from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch.object(cached_images, 'WORKER')
def test_queue_image_from_api(self, mock_worker):
    self._main_test_helper(['queue_image', 'queue_image_from_api', 'cache_image', UUID1])
    mock_worker.submit.assert_called_once_with(UUID1)