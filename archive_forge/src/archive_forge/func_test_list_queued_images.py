import sys
from unittest import mock
import fixtures
from glance.cmd import cache_manage
from glance.image_cache import client as cache_client
from glance.tests import utils as test_utils
def test_list_queued_images(self):
    self._main_test_helper(['glance.cmd.cache_manage', 'list-queued'])