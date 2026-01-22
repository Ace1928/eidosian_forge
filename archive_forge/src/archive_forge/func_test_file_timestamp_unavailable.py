from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_file_timestamp_unavailable(self):
    e = FileTimestampUnavailable('/path/foo')
    self.assertEqual('The filestamp for /path/foo is not available.', str(e))