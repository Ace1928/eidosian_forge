import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_check_contextlib_nested(self):
    f = checks.check_no_contextlib_nested
    self.assertLineFails(f, 'with contextlib.nested():', '')
    self.assertLineFails(f, '    with contextlib.nested():', '')
    self.assertLinePasses(f, '# with contextlib.nested():', '')
    self.assertLinePasses(f, 'print("with contextlib.nested():")', '')