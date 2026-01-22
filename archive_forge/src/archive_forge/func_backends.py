from .. import backend
from .._compat import properties
from . import fail
@properties.classproperty
def backends(cls):
    """
        Discover all keyrings for chaining.
        """

    def allow(keyring):
        limit = backend._limit or bool
        return not isinstance(keyring, ChainerBackend) and limit(keyring) and (keyring.priority > 0)
    allowed = filter(allow, backend.get_all_keyring())
    return sorted(allowed, key=backend.by_priority, reverse=True)