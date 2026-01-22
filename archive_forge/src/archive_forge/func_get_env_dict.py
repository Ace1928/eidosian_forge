import os
from .dependencies import ctypes
def get_env_dict(self):
    ans = {}
    _str_buf = self._envstr()
    _null = {u'\x00', b'\x00'}
    i = 0
    while _str_buf[i] not in _null:
        _str = ''
        while _str_buf[i] not in _null:
            _str += _str_buf[i]
            i += len(_str_buf[i])
            if len(_str_buf[i]) == 0:
                raise ValueError('Error processing Win32 GetEnvironmentStringsW: 0-length character encountered')
            if i > 32767:
                raise ValueError('Error processing Win32 GetEnvironmentStringsW: exceeded max environment block size (32767)')
        key, val = _str.split('=', 1)
        ans[key] = val
        i += len(_str_buf[i])
    self._free_envstr(_str_buf)
    return ans