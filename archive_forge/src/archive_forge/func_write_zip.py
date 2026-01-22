from __future__ import absolute_import, print_function, unicode_literals
import typing
import six
import tarfile
import time
import zipfile
from datetime import datetime
from .enums import ResourceType
from .errors import MissingInfoNamespace, NoSysPath
from .path import relpath
from .time import datetime_to_epoch
from .walk import Walker
def write_zip(src_fs, file, compression=zipfile.ZIP_DEFLATED, encoding='utf-8', walker=None):
    """Write the contents of a filesystem to a zip file.

    Arguments:
        src_fs (~fs.base.FS): The source filesystem to compress.
        file (str or io.IOBase): Destination file, may be a file name
            or an open file object.
        compression (int): Compression to use (one of the constants
            defined in the `zipfile` module in the stdlib). Defaults
            to `zipfile.ZIP_DEFLATED`.
        encoding (str): The encoding to use for filenames. The default
            is ``"utf-8"``, use ``"CP437"`` if compatibility with WinZip
            is desired.
        walker (~fs.walk.Walker, optional): A `Walker` instance, or `None`
            to use default walker. You can use this to specify which files
            you want to compress.

    """
    _zip = zipfile.ZipFile(file, mode='w', compression=compression, allowZip64=True)
    walker = walker or Walker()
    with _zip:
        gen_walk = walker.info(src_fs, namespaces=['details', 'stat', 'access'])
        for path, info in gen_walk:
            zip_name = relpath(path + '/' if info.is_dir else path)
            if not six.PY3:
                zip_name = zip_name.encode(encoding, 'replace')
            if info.has_namespace('stat'):
                st_mtime = info.get('stat', 'st_mtime', None)
                _mtime = time.localtime(st_mtime)
                zip_time = _mtime[0:6]
            else:
                mt = info.modified or datetime.utcnow()
                zip_time = (mt.year, mt.month, mt.day, mt.hour, mt.minute, mt.second)
            zip_info = zipfile.ZipInfo(zip_name, zip_time)
            try:
                if info.permissions is not None:
                    zip_info.external_attr = info.permissions.mode << 16
            except MissingInfoNamespace:
                pass
            if info.is_dir:
                zip_info.external_attr |= 16
                _zip.writestr(zip_info, b'')
            else:
                try:
                    sys_path = src_fs.getsyspath(path)
                except NoSysPath:
                    _zip.writestr(zip_info, src_fs.readbytes(path))
                else:
                    _zip.write(sys_path, zip_name)