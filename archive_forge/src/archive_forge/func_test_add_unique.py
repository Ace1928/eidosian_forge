import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_add_unique(self):
    """Test that adding will not duplicate ignores"""
    ignores._set_user_ignores(['foo', './bar', 'båz', 'dir1/', 'dir3\\'])
    added = ignores.add_unique_user_ignores(['xxx', './bar', 'xxx', 'dir1/', 'dir2/', 'dir3\\'])
    self.assertEqual(['xxx', 'dir2'], added)
    self.assertEqual({'foo', './bar', 'båz', 'xxx', 'dir1', 'dir2', 'dir3'}, ignores.get_user_ignores())