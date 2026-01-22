import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test___call___is_recorded(self):
    fixture = self.useFixture(FakePopen())
    proc = fixture(['foo', 'bar'], 1, None, 'in', 'out', 'err')
    self.assertEqual(1, len(fixture.procs))
    self.assertEqual(dict(args=['foo', 'bar'], bufsize=1, executable=None, stdin='in', stdout='out', stderr='err'), proc._args)