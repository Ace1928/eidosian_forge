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
def sanitize_ref_name_for_git(refname):
    """Rewrite refname so that it will be accepted by git-fast-import.
    For the detailed rules see check_ref_format.

    By rewriting the refname we are breaking uniqueness guarantees provided by bzr
    so we have to manually
    verify that resulting ref names are unique.

    :param refname: refname to rewrite
    :return: new refname
    """
    import struct
    new_refname = re.sub(b'/\\.|^\\.|\\.\\.|[' + b''.join([bytes([x]) for x in range(32)]) + b']|[\\177 ~^:?*[]|[/.]$|.lock$|@{|\\\\', b'_', refname)
    return new_refname