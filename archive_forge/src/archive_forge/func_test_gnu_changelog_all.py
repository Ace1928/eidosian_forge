import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_gnu_changelog_all(self):
    self.assertFormatterResult(log.GnuChangelogLogFormatter, 'all', b'2005-11-22  John Doe  <jdoe@example.com>, Jane Rey  <jrey@example.com>\n\n\tadd a\n\n')