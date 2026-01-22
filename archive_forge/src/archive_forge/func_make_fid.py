import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def make_fid(name, mode, userblock_size, fapl, fcpl=None, swmr=False):
    """ Get a new FileID by opening or creating a file.
    Also validates mode argument."""
    if userblock_size is not None:
        if mode in ('r', 'r+'):
            raise ValueError('User block may only be specified when creating a file')
        try:
            userblock_size = int(userblock_size)
        except (TypeError, ValueError):
            raise ValueError('User block size must be an integer')
        if fcpl is None:
            fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_userblock(userblock_size)
    if mode == 'r':
        flags = h5f.ACC_RDONLY
        if swmr and swmr_support:
            flags |= h5f.ACC_SWMR_READ
        fid = h5f.open(name, flags, fapl=fapl)
    elif mode == 'r+':
        fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
    elif mode in ['w-', 'x']:
        fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    elif mode == 'w':
        fid = h5f.create(name, h5f.ACC_TRUNC, fapl=fapl, fcpl=fcpl)
    elif mode == 'a':
        try:
            fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
        except FileNotFoundError if fapl.get_driver() in (h5fd.SEC2, h5fd.DIRECT if direct_vfd else -1, h5fd.FAMILY, h5fd.WINDOWS, h5fd.fileobj_driver, h5fd.ROS3D if ros3 else -1) else OSError:
            fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    else:
        raise ValueError('Invalid mode; must be one of r, r+, w, w-, x, a')
    try:
        if userblock_size is not None:
            existing_fcpl = fid.get_create_plist()
            if existing_fcpl.get_userblock() != userblock_size:
                raise ValueError('Requested userblock size (%d) does not match that of existing file (%d)' % (userblock_size, existing_fcpl.get_userblock()))
    except Exception as e:
        fid.close()
        raise e
    return fid