import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0, locking, page_buf_size, min_meta_keep, min_raw_keep, alignment_threshold, alignment_interval, meta_block_size, **kwds):
    """ Set up a file access property list """
    plist = h5p.create(h5p.FILE_ACCESS)
    if libver is not None:
        if libver in libver_dict:
            low = libver_dict[libver]
            high = h5f.LIBVER_LATEST
        else:
            low, high = (libver_dict[x] for x in libver)
    else:
        low, high = (h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST)
    plist.set_libver_bounds(low, high)
    plist.set_alignment(alignment_threshold, alignment_interval)
    cache_settings = list(plist.get_cache())
    if rdcc_nslots is not None:
        cache_settings[1] = rdcc_nslots
    if rdcc_nbytes is not None:
        cache_settings[2] = rdcc_nbytes
    if rdcc_w0 is not None:
        cache_settings[3] = rdcc_w0
    plist.set_cache(*cache_settings)
    if page_buf_size:
        plist.set_page_buffer_size(int(page_buf_size), int(min_meta_keep), int(min_raw_keep))
    if meta_block_size is not None:
        plist.set_meta_block_size(int(meta_block_size))
    if locking is not None:
        if hdf5_version < (1, 12, 1) and (hdf5_version[:2] != (1, 10) or hdf5_version[2] < 7):
            raise ValueError('HDF5 version >= 1.12.1 or 1.10.x >= 1.10.7 required for file locking.')
        if locking in ('false', False):
            plist.set_file_locking(False, ignore_when_disabled=False)
        elif locking in ('true', True):
            plist.set_file_locking(True, ignore_when_disabled=False)
        elif locking == 'best-effort':
            plist.set_file_locking(True, ignore_when_disabled=True)
        else:
            raise ValueError(f'Unsupported locking value: {locking}')
    if driver is None or (driver == 'windows' and sys.platform == 'win32'):
        if kwds:
            msg = "'{key}' is an invalid keyword argument for this function".format(key=next(iter(kwds)))
            raise TypeError(msg)
        return plist
    try:
        set_fapl = _drivers[driver]
    except KeyError:
        raise ValueError('Unknown driver type "%s"' % driver)
    else:
        if driver == 'ros3':
            token = kwds.pop('session_token', None)
            set_fapl(plist, **kwds)
            if token:
                if hdf5_version < (1, 14, 2):
                    raise ValueError('HDF5 >= 1.14.2 required for AWS session token')
                plist.set_fapl_ros3_token(token)
        else:
            set_fapl(plist, **kwds)
    return plist