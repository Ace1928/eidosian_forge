import doctest
import os
import sys
from io import StringIO
import breezy
from .. import bedding, crash, osutils, plugin, tests
from . import features
def test_report_bug_legacy(self):
    self.setup_fake_plugins()
    err_file = StringIO()
    try:
        raise AssertionError('my error')
    except AssertionError as e:
        crash.report_bug_legacy(sys.exc_info(), err_file)
    report = err_file.getvalue()
    for needle in ['brz: ERROR: AssertionError: my error', 'Traceback \\(most recent call last\\):', 'plugins: fake_plugin\\[1\\.2\\.3\\]']:
        self.assertContainsRe(report, needle)