import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_author_from_committer(self):
    self.rev.committer = 'John Doe <jdoe@example.com>'
    self.assertEqual('John Doe', self.lf.short_author(self.rev))