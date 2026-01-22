import os
import stat
import sys
import tempfile
import time
import zipfile
from contextlib import closing
from .. import osutils
from ..export import _export_iter_entries
from ..trace import mutter
def zip_archive_generator(tree, dest, root, subdir=None, force_mtime=None, recurse_nested=False):
    """ Export this tree to a new zip file.

    `dest` will be created holding the contents of this tree; if it
    already exists, it will be overwritten".
    """
    compression = zipfile.ZIP_DEFLATED
    with tempfile.SpooledTemporaryFile() as buf:
        with closing(zipfile.ZipFile(buf, 'w', compression)) as zipf, tree.lock_read():
            for dp, tp, ie in _export_iter_entries(tree, subdir, recurse_nested=recurse_nested):
                mutter('  export {%s} kind %s to %s', tp, ie.kind, dest)
                if force_mtime is not None:
                    mtime = force_mtime
                else:
                    mtime = tree.get_file_mtime(tp)
                date_time = time.localtime(mtime)[:6]
                filename = osutils.pathjoin(root, dp)
                if ie.kind == 'file':
                    zinfo = zipfile.ZipInfo(filename=filename, date_time=date_time)
                    zinfo.compress_type = compression
                    zinfo.external_attr = _FILE_ATTR
                    content = tree.get_file_text(tp)
                    zipf.writestr(zinfo, content)
                elif ie.kind in ('directory', 'tree-reference'):
                    zinfo = zipfile.ZipInfo(filename=filename + '/', date_time=date_time)
                    zinfo.compress_type = compression
                    zinfo.external_attr = _DIR_ATTR
                    zipf.writestr(zinfo, '')
                elif ie.kind == 'symlink':
                    zinfo = zipfile.ZipInfo(filename=filename + '.lnk', date_time=date_time)
                    zinfo.compress_type = compression
                    zinfo.external_attr = _FILE_ATTR
                    zipf.writestr(zinfo, tree.get_symlink_target(tp))
        buf.seek(0)
        yield from osutils.file_iterator(buf)