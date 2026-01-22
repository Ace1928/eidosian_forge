import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_installs_restores_global(self):
    fixture = FakePopen()
    popen = subprocess.Popen
    fixture.setUp()
    try:
        self.assertEqual(subprocess.Popen, fixture)
    finally:
        fixture.cleanUp()
        self.assertEqual(subprocess.Popen, popen)