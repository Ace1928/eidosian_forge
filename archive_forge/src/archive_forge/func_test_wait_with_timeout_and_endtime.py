import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_wait_with_timeout_and_endtime(self):
    proc = FakeProcess({}, {})
    self.assertEqual(0, proc.wait(timeout=4, endtime=7))