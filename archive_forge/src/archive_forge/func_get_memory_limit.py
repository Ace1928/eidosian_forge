import binascii
import lzma
import platform
import sys
def get_memory_limit():
    """
    Get memory limit for allocating decompression chunk buffer.
    :return: allowed chunk size in bytes.
    """
    default_limit = int(128000000.0)
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin') or sys.platform.startswith('openbsd'):
        import resource
        import psutil
        try:
            soft, _ = resource.getrlimit(resource.RLIMIT_DATA)
            if soft == -1:
                avmem = psutil.virtual_memory().available
                return min(default_limit, avmem - int(256000000.0) >> 2)
            else:
                return min(default_limit, soft - int(256000000.0) >> 2)
        except AttributeError:
            pass
    return default_limit