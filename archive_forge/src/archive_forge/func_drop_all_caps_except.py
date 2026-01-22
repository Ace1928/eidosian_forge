import enum
import os
import platform
import sys
import cffi
def drop_all_caps_except(effective, permitted, inheritable):
    """Set (effective, permitted, inheritable) to provided list of caps"""
    eff = _caps_to_mask(effective)
    prm = _caps_to_mask(permitted)
    inh = _caps_to_mask(inheritable)
    header = ffi.new('cap_user_header_t', {'version': crt._LINUX_CAPABILITY_VERSION_2, 'pid': 0})
    data = ffi.new('struct __user_cap_data_struct[2]')
    data[0].effective = eff & 4294967295
    data[1].effective = eff >> 32
    data[0].permitted = prm & 4294967295
    data[1].permitted = prm >> 32
    data[0].inheritable = inh & 4294967295
    data[1].inheritable = inh >> 32
    ret = _capset(header, data)
    if ret != 0:
        errno = ffi.errno
        raise OSError(errno, os.strerror(errno))