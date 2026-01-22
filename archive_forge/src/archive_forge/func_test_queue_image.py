import sys
from unittest import mock
import fixtures
from glance.cmd import cache_manage
from glance.image_cache import client as cache_client
from glance.tests import utils as test_utils
@mock.patch.object(cache_manage, 'user_confirm')
def test_queue_image(self, mock_user_confirm):
    self._main_test_helper(['glance.cmd.cache_manage', 'queue-image', UUID1])
    self.assertEqual(1, mock_user_confirm.call_count)