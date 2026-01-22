import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_inject_content_stdout(self):

    def get_info(args):
        return {'stdout': 'stdout'}
    fixture = self.useFixture(FakePopen(get_info))
    proc = fixture(['foo'])
    self.assertEqual('stdout', proc.stdout)