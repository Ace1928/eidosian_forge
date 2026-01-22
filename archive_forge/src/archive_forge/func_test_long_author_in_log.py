import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_long_author_in_log(self):
    """Log includes the author name if it's set in
        the revision properties
        """
    wt = self.make_standard_commit('test_author_log')
    self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\nauthor: John Doe <jdoe@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_author_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=1))