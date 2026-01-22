import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_custom_returncode(self):

    def get_info(proc_args):
        return dict(returncode=1)
    proc = self.useFixture(FakePopen(get_info))(['foo'])
    self.assertEqual(None, proc.returncode)
    self.assertEqual(1, proc.wait())
    self.assertEqual(1, proc.returncode)