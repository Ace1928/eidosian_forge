from .base import BaseCompress
from typing import Union
def ensure_zstd_available():
    """
    Ensure Zstd is available
    """
    global _zstd_available, zstd
    if _zstd_available is False:
        from lazyops.utils.imports import resolve_missing
        resolve_missing('zstd', required=True)
        import zstd
        _zstd_available = True
        globals()['zstd'] = zstd