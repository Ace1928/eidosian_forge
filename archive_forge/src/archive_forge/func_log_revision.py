import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def log_revision(self, revision):
    self.revisions.append(revision)