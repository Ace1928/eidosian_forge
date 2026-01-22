import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def geosearch(self, name: KeyT, member: Union[FieldT, None]=None, longitude: Union[float, None]=None, latitude: Union[float, None]=None, unit: str='m', radius: Union[float, None]=None, width: Union[float, None]=None, height: Union[float, None]=None, sort: Union[str, None]=None, count: Union[int, None]=None, any: bool=False, withcoord: bool=False, withdist: bool=False, withhash: bool=False) -> ResponseT:
    """
        Return the members of specified key identified by the
        ``name`` argument, which are within the borders of the
        area specified by a given shape. This command extends the
        GEORADIUS command, so in addition to searching within circular
        areas, it supports searching within rectangular areas.

        This command should be used in place of the deprecated
        GEORADIUS and GEORADIUSBYMEMBER commands.

        ``member`` Use the position of the given existing
         member in the sorted set. Can't be given with ``longitude``
         and ``latitude``.

        ``longitude`` and ``latitude`` Use the position given by
        this coordinates. Can't be given with ``member``
        ``radius`` Similar to GEORADIUS, search inside circular
        area according the given radius. Can't be given with
        ``height`` and ``width``.
        ``height`` and ``width`` Search inside an axis-aligned
        rectangle, determined by the given height and width.
        Can't be given with ``radius``

        ``unit`` must be one of the following : m, km, mi, ft.
        `m` for meters (the default value), `km` for kilometers,
        `mi` for miles and `ft` for feet.

        ``sort`` indicates to return the places in a sorted way,
        ASC for nearest to furthest and DESC for furthest to nearest.

        ``count`` limit the results to the first count matching items.

        ``any`` is set to True, the command will return as soon as
        enough matches are found. Can't be provided without ``count``

        ``withdist`` indicates to return the distances of each place.
        ``withcoord`` indicates to return the latitude and longitude of
        each place.

        ``withhash`` indicates to return the geohash string of each place.

        For more information see https://redis.io/commands/geosearch
        """
    return self._geosearchgeneric('GEOSEARCH', name, member=member, longitude=longitude, latitude=latitude, unit=unit, radius=radius, width=width, height=height, sort=sort, count=count, any=any, withcoord=withcoord, withdist=withdist, withhash=withhash, store=None, store_dist=None)