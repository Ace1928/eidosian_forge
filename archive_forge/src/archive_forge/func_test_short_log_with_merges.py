import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_log_with_merges(self):
    wt = self._prepare_tree_with_merges()
    self.assertFormatterResult(b'    2 Joe Foo\t2005-11-22 [merge]\n      rev-2\n\n    1 Joe Foo\t2005-11-22\n      rev-1\n\n', wt.branch, log.ShortLogFormatter)