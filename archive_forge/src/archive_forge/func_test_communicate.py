import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_communicate(self):
    proc = FakeProcess({}, {})
    self.assertEqual(('', ''), proc.communicate())
    self.assertEqual(0, proc.returncode)