import os
import sys
import tarfile
from contextlib import closing
from io import BytesIO
from .. import errors, osutils
from ..export import _export_iter_entries
def tarball_generator(tree, root, subdir=None, force_mtime=None, format='', recurse_nested=False):
    """Export tree contents to a tarball.

    Args:
      tree: Tree to export
      subdir: Sub directory to export
      force_mtime: Option mtime to force, instead of using tree
        timestamps.
    Returns: A generator that will produce file content chunks.
    """
    buf = BytesIO()
    with closing(tarfile.open(None, 'w:%s' % format, buf)) as ball, tree.lock_read():
        for final_path, tree_path, entry in _export_iter_entries(tree, subdir, recurse_nested=recurse_nested):
            item, fileobj = prepare_tarball_item(tree, root, final_path, tree_path, entry, force_mtime)
            ball.addfile(item, fileobj)
            yield buf.getvalue()
            buf.truncate(0)
            buf.seek(0)
    yield buf.getvalue()