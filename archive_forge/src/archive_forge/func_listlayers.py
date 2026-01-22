import glob
import logging
import os
from pathlib import Path
import platform
import warnings
from fiona._env import (
from fiona._env import driver_count
from fiona._show_versions import show_versions
from fiona.collection import BytesCollection, Collection
from fiona.drvsupport import supported_drivers
from fiona.env import ensure_env_with_credentials, Env
from fiona.errors import FionaDeprecationWarning
from fiona.io import MemoryFile
from fiona.model import Feature, Geometry, Properties
from fiona.ogrext import (
from fiona.path import ParsedPath, parse_path, vsi_path
from fiona.vfs import parse_paths as vfs_parse_paths
from fiona import _geometry, _err, rfc3339
import uuid
@ensure_env_with_credentials
def listlayers(fp, vfs=None, **kwargs):
    """Lists the layers (collections) in a dataset.

    Archive files must be prefixed like "zip://" or "tar://".

    Parameters
    ----------
    fp : str, pathlib.Path, or file-like object
        A dataset identifier or file object containing a dataset.
    vfs : str
        This is a deprecated parameter. A URI scheme such as "zip://"
        should be used instead.
    kwargs : dict
        Dataset opening options and other keyword args.

    Returns
    -------
    list of str
        A list of layer name strings.

    Raises
    ------
    TypeError
        If the input is not a str, Path, or file object.

    """
    if hasattr(fp, 'read'):
        with MemoryFile(fp.read()) as memfile:
            return _listlayers(memfile.name, **kwargs)
    else:
        if isinstance(fp, Path):
            fp = str(fp)
        if not isinstance(fp, str):
            raise TypeError('invalid path: %r' % fp)
        if vfs and (not isinstance(vfs, str)):
            raise TypeError('invalid vfs: %r' % vfs)
        if vfs:
            warnings.warn('The vfs keyword argument is deprecated and will be removed in 2.0. Instead, pass a URL that uses a zip or tar (for example) scheme.', FionaDeprecationWarning, stacklevel=2)
            pobj_vfs = parse_path(vfs)
            pobj_path = parse_path(fp)
            pobj = ParsedPath(pobj_path.path, pobj_vfs.path, pobj_vfs.scheme)
        else:
            pobj = parse_path(fp)
        return _listlayers(vsi_path(pobj), **kwargs)