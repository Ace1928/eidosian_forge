import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_default(self):
    self.assertFormatterResult(log.ShortLogFormatter, None, b'    1 John Doe\t2005-11-22\n      add a\n\n')