from __future__ import annotations
import warnings
from nbformat._struct import Struct
def cast_str(obj):
    """Cast an object as a string."""
    if isinstance(obj, bytes):
        warnings.warn('A notebook got bytes instead of likely base64 encoded values.The content will likely be corrupted.', UserWarning, stacklevel=3)
        return obj.decode('ascii', 'replace')
    if not isinstance(obj, str):
        raise AssertionError
    return obj