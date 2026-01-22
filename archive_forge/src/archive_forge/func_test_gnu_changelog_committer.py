import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_gnu_changelog_committer(self):
    self.assertFormatterResult(log.GnuChangelogLogFormatter, 'committer', b'2005-11-22  Lorem Ipsum  <test@example.com>\n\n\tadd a\n\n')