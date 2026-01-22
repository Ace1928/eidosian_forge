import doctest
import os
import sys
from io import StringIO
import breezy
from .. import bedding, crash, osutils, plugin, tests
from . import features
def test_apport_report(self):
    crash_dir = osutils.joinpath((self.test_base_dir, 'crash'))
    os.mkdir(crash_dir)
    self.overrideEnv('APPORT_CRASH_DIR', crash_dir)
    self.assertEqual(crash_dir, bedding.crash_dir())
    self.overrideAttr(breezy.get_global_state(), 'plugin_warnings', {'example': ['Failed to load plugin foo']})
    stderr = StringIO()
    try:
        raise AssertionError('my error')
    except AssertionError as e:
        crash_filename = crash.report_bug_to_apport(sys.exc_info(), stderr)
    self.assertContainsRe(stderr.getvalue(), '    apport-bug %s' % crash_filename)
    with open(crash_filename) as crash_file:
        report = crash_file.read()
    self.assertContainsRe(report, '(?m)^BrzVersion:')
    self.assertContainsRe(report, 'my error')
    self.assertContainsRe(report, 'AssertionError')
    self.assertContainsRe(report, 'ExecutablePath')
    self.assertContainsRe(report, 'test_apport_report')
    self.assertContainsRe(report, '(?m)^CommandLine:')
    self.assertContainsRe(report, 'Failed to load plugin foo')