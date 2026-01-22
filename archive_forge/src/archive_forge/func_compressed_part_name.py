import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
def compressed_part_name(basename):
    candidates = ['%s.%s' % (basename, ext) for ext in PART_EXTS]
    if basename in (DATA_PART, CTRL_PART):
        candidates.append(basename)
    parts = actual_names.intersection(set(candidates))
    if not parts:
        raise DebError('missing required part in given .deb (expected one of: %s)' % candidates)
    if len(parts) > 1:
        raise DebError('too many parts in given .deb (was looking for only one of: %s)' % candidates)
    return list(parts)[0]