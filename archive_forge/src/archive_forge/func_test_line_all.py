import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_all(self):
    self.assertFormatterResult(log.LineLogFormatter, 'all', b'1: John Doe, Jane Rey 2005-11-22 add a\n')