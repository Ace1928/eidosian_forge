import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_long_verbose_log(self):
    """Verbose log includes changed files

        bug #4676
        """
    wt = self.make_standard_commit('test_long_verbose_log', authors=[])
    self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_long_verbose_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\nadded:\n  a\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=1), show_log_kwargs=dict(verbose=True))