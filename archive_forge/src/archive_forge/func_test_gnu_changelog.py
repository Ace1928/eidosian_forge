import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_gnu_changelog(self):
    wt = self.make_standard_commit('nicky', authors=[])
    self.assertFormatterResult(b'2005-11-22  Lorem Ipsum  <test@example.com>\n\n\tadd a\n\n', wt.branch, log.GnuChangelogLogFormatter)