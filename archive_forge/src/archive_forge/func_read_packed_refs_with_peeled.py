import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def read_packed_refs_with_peeled(f):
    """Read a packed refs file including peeled refs.

    Assumes the "# pack-refs with: peeled" line was already read. Yields tuples
    with ref names, SHA1s, and peeled SHA1s (or None).

    Args:
      f: file-like object to read from, seek'ed to the second line
    """
    last = None
    for line in f:
        if line[0] == b'#':
            continue
        line = line.rstrip(b'\r\n')
        if line.startswith(b'^'):
            if not last:
                raise PackedRefsException('unexpected peeled ref line')
            if not valid_hexsha(line[1:]):
                raise PackedRefsException('Invalid hex sha %r' % line[1:])
            sha, name = _split_ref_line(last)
            last = None
            yield (sha, name, line[1:])
        else:
            if last:
                sha, name = _split_ref_line(last)
                yield (sha, name, None)
            last = line
    if last:
        sha, name = _split_ref_line(last)
        yield (sha, name, None)