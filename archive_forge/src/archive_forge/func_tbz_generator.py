import os
import sys
import tarfile
from contextlib import closing
from io import BytesIO
from .. import errors, osutils
from ..export import _export_iter_entries
def tbz_generator(tree, dest, root, subdir, force_mtime=None, recurse_nested=False):
    """Export this tree to a new tar file.

    `dest` will be created holding the contents of this tree; if it
    already exists, it will be clobbered, like with "tar -c".
    """
    return tarball_generator(tree, root, subdir, force_mtime, format='bz2', recurse_nested=recurse_nested)