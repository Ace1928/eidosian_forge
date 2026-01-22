import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def checkDelta(self, delta, **kw):
    """Check the filenames touched by a delta are as expected.

        Caller only have to pass in the list of files for each part, all
        unspecified parts are considered empty (and checked as such).
        """
    for n in ('added', 'removed', 'renamed', 'modified', 'unchanged'):
        expected = kw.get(n, [])
        got = [x.path[1] or x.path[0] for x in getattr(delta, n)]
        self.assertEqual(expected, got)