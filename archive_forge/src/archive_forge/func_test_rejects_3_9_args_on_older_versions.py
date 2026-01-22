import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
@testtools.skipUnless(sys.version_info < (3, 9), 'only relevant on Python <3.9')
def test_rejects_3_9_args_on_older_versions(self):
    fixture = self.useFixture(FakePopen(lambda proc_args: {}))
    for arg_name in ('group', 'extra_groups', 'user', 'umask'):
        kwargs = {arg_name: arg_name}
        expected_message = ".* got an unexpected keyword argument '{}'".format(arg_name)
        with testtools.ExpectedException(TypeError, expected_message):
            fixture(args='args', **kwargs)