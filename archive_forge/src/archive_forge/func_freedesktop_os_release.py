import collections
import os
import re
import sys
import functools
import itertools
def freedesktop_os_release():
    """Return operation system identification from freedesktop.org os-release
    """
    global _os_release_cache
    if _os_release_cache is None:
        errno = None
        for candidate in _os_release_candidates:
            try:
                with open(candidate, encoding='utf-8') as f:
                    _os_release_cache = _parse_os_release(f)
                break
            except OSError as e:
                errno = e.errno
        else:
            raise OSError(errno, f'Unable to read files {', '.join(_os_release_candidates)}')
    return _os_release_cache.copy()