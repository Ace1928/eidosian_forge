import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_with_authors(self):
    wt = self.make_standard_commit('nicky', authors=['Fooa Fooz <foo@example.com>', 'Bari Baro <bar@example.com>'])
    self.assertFormatterResult(b'2005-11-22  Fooa Fooz  <foo@example.com>\n\n\tadd a\n\n', wt.branch, log.GnuChangelogLogFormatter)