import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_communicate_with_input(self):
    proc = FakeProcess({}, {'stdout': io.BytesIO(b'foo')})
    self.assertEqual((b'foo', ''), proc.communicate(input=b'bar'))