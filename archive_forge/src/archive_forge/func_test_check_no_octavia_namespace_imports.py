import testtools
from oslotest import base
from octavia_lib.hacking import checks
def test_check_no_octavia_namespace_imports(self):
    f = checks.check_no_octavia_namespace_imports
    self.assertLinePasses(f, 'from octavia_lib import constants')
    self.assertLinePasses(f, 'import octavia_lib.constants')
    self.assertLineFails(f, 'from octavia.common import rpc')
    self.assertLineFails(f, 'from octavia import context')
    self.assertLineFails(f, 'import octavia.common.config')