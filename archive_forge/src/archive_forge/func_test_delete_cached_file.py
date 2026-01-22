import os
from unittest import mock
import ddt
from glance.image_cache.drivers import sqlite
from glance.tests import utils
@ddt.data(True, False)
def test_delete_cached_file(self, throw_not_exists):
    with mock.patch.object(os, 'unlink') as mock_unlink:
        if throw_not_exists:
            mock_unlink.side_effect = OSError((2, 'File not found'))
    sqlite.delete_cached_file('/tmp/dummy_file')