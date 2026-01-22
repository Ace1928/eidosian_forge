import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def gen_diff_header(paths, modes, shas):
    """Write a blob diff header.

    Args:
      paths: Tuple with old and new path
      modes: Tuple with old and new modes
      shas: Tuple with old and new shas
    """
    old_path, new_path = paths
    old_mode, new_mode = modes
    old_sha, new_sha = shas
    if old_path is None and new_path is not None:
        old_path = new_path
    if new_path is None and old_path is not None:
        new_path = old_path
    old_path = patch_filename(old_path, b'a')
    new_path = patch_filename(new_path, b'b')
    yield (b'diff --git ' + old_path + b' ' + new_path + b'\n')
    if old_mode != new_mode:
        if new_mode is not None:
            if old_mode is not None:
                yield ('old file mode %o\n' % old_mode).encode('ascii')
            yield ('new file mode %o\n' % new_mode).encode('ascii')
        else:
            yield ('deleted file mode %o\n' % old_mode).encode('ascii')
    yield (b'index ' + shortid(old_sha) + b'..' + shortid(new_sha))
    if new_mode is not None and old_mode is not None:
        yield (' %o' % new_mode).encode('ascii')
    yield b'\n'