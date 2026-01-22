import configparser
import glob
import os
import sys
from os.path import join as pjoin
from packaging.version import Version
from .environment import get_nipy_system_dir, get_nipy_user_dir
def make_datasource(pkg_def, **kwargs):
    """Return datasource defined by `pkg_def` as found in `data_path`

    `data_path` is the only allowed keyword argument.

    `pkg_def` is a dictionary with at least one key - 'relpath'.  'relpath' is
    a relative path with unix forward slash separators.

    The relative path to the data is found with::

        names = pkg_def['name'].split('/')
        rel_path = os.path.join(names)

    We search for this relative path in the list of paths given by `data_path`.
    By default `data_path` is given by ``get_data_path()`` in this module.

    If we can't find the relative path, raise a DataError

    Parameters
    ----------
    pkg_def : dict
       dict containing at least the key 'relpath'. 'relpath' is the data path
       of the package relative to `data_path`.  It is in unix path format
       (using forward slashes as directory separators).  `pkg_def` can also
       contain optional keys 'name' (the name of the package), and / or a key
       'install hint' that we use in the returned error message from trying to
       use the resulting datasource
    data_path : sequence of strings or None, optional
       sequence of paths in which to search for data.  If None (the
       default), then use ``get_data_path()``

    Returns
    -------
    datasource : ``VersionedDatasource``
       An initialized ``VersionedDatasource`` instance
    """
    if any((key for key in kwargs if key != 'data_path')):
        raise ValueError('Unexpected keyword argument(s)')
    data_path = kwargs.get('data_path')
    if data_path is None:
        data_path = get_data_path()
    unix_relpath = pkg_def['relpath']
    names = unix_relpath.split('/')
    try:
        pth = find_data_dir(data_path, *names)
    except DataError as e:
        pth = [pjoin(this_data_path, *names) for this_data_path in data_path]
        pkg_hint = pkg_def.get('install hint', DEFAULT_INSTALL_HINT)
        msg = f'{e}; Is it possible you have not installed a data package?'
        if 'name' in pkg_def:
            msg += f'\n\nYou may need the package "{pkg_def['name']}"'
        if pkg_hint is not None:
            msg += f'\n\n{pkg_hint}'
        raise DataError(msg)
    return VersionedDatasource(pth)