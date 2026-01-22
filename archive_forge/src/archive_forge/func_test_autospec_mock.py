import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_autospec_mock(self):

    class A(object):

        class B(object):
            C = None
    with mock.patch.object(A, 'B'):
        with self.assertRaisesRegex(InvalidSpecError, "Cannot autospec attr 'B' from target <MagicMock spec='A'"):
            create_autospec(A).B
        with self.assertRaisesRegex(InvalidSpecError, "Cannot autospec attr 'B' from target 'A'"):
            mock.patch.object(A, 'B', autospec=True).start()
        with self.assertRaisesRegex(InvalidSpecError, "Cannot autospec attr 'C' as the patch target "):
            mock.patch.object(A.B, 'C', autospec=True).start()
        with self.assertRaisesRegex(InvalidSpecError, "Cannot spec attr 'B' as the spec "):
            mock.patch.object(A, 'B', spec=A.B).start()
        with self.assertRaisesRegex(InvalidSpecError, "Cannot spec attr 'B' as the spec_set "):
            mock.patch.object(A, 'B', spec_set=A.B).start()
        with self.assertRaisesRegex(InvalidSpecError, "Cannot spec attr 'B' as the spec_set "):
            mock.patch.object(A, 'B', spec_set=A.B).start()
        with self.assertRaisesRegex(InvalidSpecError, 'Cannot spec a Mock object.'):
            mock.Mock(A.B)
        with mock.patch('builtins.open', mock.mock_open()):
            mock.mock_open()