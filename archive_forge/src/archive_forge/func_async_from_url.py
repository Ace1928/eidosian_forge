from typing import TYPE_CHECKING
def async_from_url(url, **kwargs):
    """
    Returns an active AsyncKeyDB client generated from the given database URL.

    Will attempt to extract the database id from the path url fragment, if
    none is provided.
    """
    from aiokeydb.v1.asyncio.core import AsyncKeyDB
    return AsyncKeyDB.from_url(url, **kwargs)