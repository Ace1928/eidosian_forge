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
def get_executable_change(old_is_x, new_is_x):
    descr = {True: b'+x', False: b'-x', None: b'??'}
    if old_is_x != new_is_x:
        return [b'%s to %s' % (descr[old_is_x], descr[new_is_x])]
    else:
        return []