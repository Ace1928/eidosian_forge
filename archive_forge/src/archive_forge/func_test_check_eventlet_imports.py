import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_check_eventlet_imports(self):
    f = checks.check_no_eventlet_imports
    self.assertLineFails(f, 'import eventlet')
    self.assertLineFails(f, 'import eventlet.timeout')
    self.assertLineFails(f, 'from eventlet import timeout')
    self.assertLineFails(f, 'from eventlet.timeout import Timeout')
    self.assertLineFails(f, 'from eventlet.timeout import (Timeout, X)')
    self.assertLinePasses(f, 'import is.not.eventlet')
    self.assertLinePasses(f, 'from is.not.eventlet')
    self.assertLinePasses(f, 'from mymod import eventlet')
    self.assertLinePasses(f, 'from mymod.eventlet import amod')
    self.assertLinePasses(f, 'print("eventlet not here")')
    self.assertLinePasses(f, 'print("eventlet.timeout")')
    self.assertLinePasses(f, 'from mymod.timeout import (eventlet, X)')