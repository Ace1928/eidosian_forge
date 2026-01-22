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
def external_diff(old_label, oldlines, new_label, newlines, to_file, diff_opts):
    """Display a diff by calling out to the external diff program."""
    import tempfile
    to_file.flush()
    oldtmp_fd, old_abspath = tempfile.mkstemp(prefix='brz-diff-old-')
    newtmp_fd, new_abspath = tempfile.mkstemp(prefix='brz-diff-new-')
    oldtmpf = os.fdopen(oldtmp_fd, 'wb')
    newtmpf = os.fdopen(newtmp_fd, 'wb')
    try:
        oldtmpf.writelines(oldlines)
        newtmpf.writelines(newlines)
        oldtmpf.close()
        newtmpf.close()
        if not diff_opts:
            diff_opts = []
        if sys.platform == 'win32':
            old_label = old_label.encode('mbcs')
            new_label = new_label.encode('mbcs')
        diffcmd = ['diff', '--label', old_label, old_abspath, '--label', new_label, new_abspath, '--binary']
        diff_opts = default_style_unified(diff_opts)
        if diff_opts:
            diffcmd.extend(diff_opts)
        pipe = _spawn_external_diff(diffcmd, capture_errors=True)
        out, err = pipe.communicate()
        rc = pipe.returncode
        out += b'\n'
        if rc == 2:
            lang_c_out = out
            pipe = _spawn_external_diff(diffcmd, capture_errors=False)
            out, err = pipe.communicate()
            to_file.write(out + b'\n')
            if pipe.returncode != 2:
                raise errors.BzrError('external diff failed with exit code 2 when run with LANG=C and LC_ALL=C, but not when run natively: %r' % (diffcmd,))
            first_line = lang_c_out.split(b'\n', 1)[0]
            m = re.match(b'^(binary )?files.*differ$', first_line, re.I)
            if m is None:
                raise errors.BzrError('external diff failed with exit code 2; command: %r' % (diffcmd,))
            else:
                return
        to_file.write(out)
        if rc not in (0, 1):
            if rc < 0:
                msg = 'signal %d' % -rc
            else:
                msg = 'exit code %d' % rc
            raise errors.BzrError('external diff failed with %s; command: %r' % (msg, diffcmd))
    finally:
        oldtmpf.close()
        newtmpf.close()

        def cleanup(path):
            try:
                os.remove(path)
            except OSError as e:
                if e.errno not in (errno.ENOENT,):
                    warning('Failed to delete temporary file: %s %s', path, e)
        cleanup(old_abspath)
        cleanup(new_abspath)