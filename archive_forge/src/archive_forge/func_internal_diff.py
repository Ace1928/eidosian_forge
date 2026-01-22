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
def internal_diff(old_label, oldlines, new_label, newlines, to_file, allow_binary=False, sequence_matcher=None, path_encoding='utf8', context_lines=DEFAULT_CONTEXT_AMOUNT):
    if allow_binary is False:
        textfile.check_text_lines(oldlines)
        textfile.check_text_lines(newlines)
    if sequence_matcher is None:
        sequence_matcher = patiencediff.PatienceSequenceMatcher
    ud = unified_diff_bytes(oldlines, newlines, fromfile=old_label.encode(path_encoding, 'replace'), tofile=new_label.encode(path_encoding, 'replace'), n=context_lines, sequencematcher=sequence_matcher)
    ud = list(ud)
    if len(ud) == 0:
        return
    if not oldlines:
        ud[2] = ud[2].replace(b'-1,0', b'-0,0')
    elif not newlines:
        ud[2] = ud[2].replace(b'+1,0', b'+0,0')
    for line in ud:
        to_file.write(line)
        if not line.endswith(b'\n'):
            to_file.write(b'\n\\ No newline at end of file\n')
    to_file.write(b'\n')