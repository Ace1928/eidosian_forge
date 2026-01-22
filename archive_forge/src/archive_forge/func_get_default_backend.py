from typing import Dict
import dns.exception
from dns._asyncbackend import (  # noqa: F401  lgtm[py/unused-import]
def get_default_backend() -> Backend:
    """Get the default backend, initializing it if necessary."""
    if _default_backend:
        return _default_backend
    return set_default_backend(sniff())