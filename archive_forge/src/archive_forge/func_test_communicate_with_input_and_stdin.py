import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_communicate_with_input_and_stdin(self):
    stdin = io.BytesIO()
    proc = FakeProcess({}, {'stdin': stdin})
    proc.communicate(input=b'hello')
    self.assertEqual(b'hello', stdin.getvalue())