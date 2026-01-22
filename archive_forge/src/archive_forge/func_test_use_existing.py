import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_use_existing(self):
    patterns = ['*.o', '*.py[co]', 'å*']
    ignores._set_user_ignores(patterns)
    user_ignores = ignores.get_user_ignores()
    self.assertEqual(set(patterns), user_ignores)