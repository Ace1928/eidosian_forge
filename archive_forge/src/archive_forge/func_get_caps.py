import enum
import os
import platform
import sys
import cffi
def get_caps():
    """Return (effective, permitted, inheritable) as lists of caps"""
    header = ffi.new('cap_user_header_t', {'version': crt._LINUX_CAPABILITY_VERSION_2, 'pid': 0})
    data = ffi.new('struct __user_cap_data_struct[2]')
    ret = _capget(header, data)
    if ret != 0:
        errno = ffi.errno
        raise OSError(errno, os.strerror(errno))
    return (_mask_to_caps(data[0].effective | data[1].effective << 32), _mask_to_caps(data[0].permitted | data[1].permitted << 32), _mask_to_caps(data[0].inheritable | data[1].inheritable << 32))