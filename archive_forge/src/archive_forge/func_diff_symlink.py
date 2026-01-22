import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
def diff_symlink(self, old_target, new_target):
    if old_target is None:
        self.to_file.write(b"=== target is '%s'\n" % new_target.encode(self.path_encoding, 'replace'))
    elif new_target is None:
        self.to_file.write(b"=== target was '%s'\n" % old_target.encode(self.path_encoding, 'replace'))
    else:
        self.to_file.write(b"=== target changed '%s' => '%s'\n" % (old_target.encode(self.path_encoding, 'replace'), new_target.encode(self.path_encoding, 'replace')))
    return self.CHANGED