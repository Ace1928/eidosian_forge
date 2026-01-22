from typing import Dict
import dns.exception
from dns._asyncbackend import (  # noqa: F401  lgtm[py/unused-import]
def set_default_backend(name: str) -> Backend:
    """Set the default backend.

    It's not normally necessary to call this method, as
    ``get_default_backend()`` will initialize the backend
    appropriately in many cases.  If ``sniffio`` is not installed, or
    in testing situations, this function allows the backend to be set
    explicitly.
    """
    global _default_backend
    _default_backend = get_backend(name)
    return _default_backend