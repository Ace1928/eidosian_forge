from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_resize_volume_with_id(self):
    self._test_resize_volume(152, 152)