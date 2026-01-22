import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def is_empty_dir(self, tree, path):
    try:
        if tree.kind(path) != 'directory':
            return False
    except _mod_transport.NoSuchFile:
        self.warning('Skipping empty_dir detection - no file_id for %s' % (path,))
        return False
    contents = list(tree.walkdirs(prefix=path))[0]
    if len(contents[1]) == 0:
        return True
    else:
        return False