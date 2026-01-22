import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_with_popen_custom(self):
    self.useFixture(FakePopen())
    with subprocess.Popen(['ls -lh']) as proc:
        self.assertEqual(None, proc.returncode)
        self.assertEqual(['ls -lh'], proc.args)