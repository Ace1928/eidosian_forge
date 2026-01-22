import zipfile
import tarfile
import os
import shutil
import posixpath
import contextlib
from distutils.errors import DistutilsError
from ._path import ensure_directory
def unpack_tarfile(filename, extract_dir, progress_filter=default_filter):
    """Unpack tar/tar.gz/tar.bz2 `filename` to `extract_dir`

    Raises ``UnrecognizedFormat`` if `filename` is not a tarfile (as determined
    by ``tarfile.open()``).  See ``unpack_archive()`` for an explanation
    of the `progress_filter` argument.
    """
    try:
        tarobj = tarfile.open(filename)
    except tarfile.TarError as e:
        raise UnrecognizedFormat('%s is not a compressed or uncompressed tar file' % (filename,)) from e
    for member, final_dst in _iter_open_tar(tarobj, extract_dir, progress_filter):
        try:
            tarobj._extract_member(member, final_dst)
        except tarfile.ExtractError:
            pass
    return True