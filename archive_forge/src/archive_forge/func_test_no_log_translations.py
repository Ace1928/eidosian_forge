import os
import textwrap
from unittest import mock
import pycodestyle
from os_win._hacking import checks
from os_win.tests.unit import test_base
def test_no_log_translations(self):
    for log in checks._all_log_levels:
        bad = 'LOG.%s(_("Bad"))' % log
        self.assertEqual(1, len(list(checks.no_translate_logs(bad))))
        bad = 'LOG.%s(_(msg))' % log
        self.assertEqual(1, len(list(checks.no_translate_logs(bad))))