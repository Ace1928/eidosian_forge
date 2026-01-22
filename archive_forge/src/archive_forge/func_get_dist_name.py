import importlib.metadata
from typing import Any, Optional, Protocol, cast
def get_dist_name(dist: importlib.metadata.Distribution) -> str:
    """Get the distribution's project name.

    The ``name`` attribute is only available in Python 3.10 or later. We are
    targeting exactly that, but Mypy does not know this.
    """
    name = cast(Any, dist).name
    if not isinstance(name, str):
        raise BadMetadata(dist, reason="invalid metadata entry 'name'")
    return name