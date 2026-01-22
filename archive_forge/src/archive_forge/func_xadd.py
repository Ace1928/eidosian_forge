import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xadd(self, name: KeyT, fields: Dict[FieldT, EncodableT], id: StreamIdT='*', maxlen: Union[int, None]=None, approximate: bool=True, nomkstream: bool=False, minid: Union[StreamIdT, None]=None, limit: Union[int, None]=None) -> ResponseT:
    """
        Add to a stream.
        name: name of the stream
        fields: dict of field/value pairs to insert into the stream
        id: Location to insert this record. By default it is appended.
        maxlen: truncate old stream members beyond this size.
        Can't be specified with minid.
        approximate: actual stream length may be slightly more than maxlen
        nomkstream: When set to true, do not make a stream
        minid: the minimum id in the stream to query.
        Can't be specified with maxlen.
        limit: specifies the maximum number of entries to retrieve

        For more information see https://redis.io/commands/xadd
        """
    pieces: list[EncodableT] = []
    if maxlen is not None and minid is not None:
        raise DataError('Only one of ```maxlen``` or ```minid``` may be specified')
    if maxlen is not None:
        if not isinstance(maxlen, int) or maxlen < 0:
            raise DataError('XADD maxlen must be non-negative integer')
        pieces.append(b'MAXLEN')
        if approximate:
            pieces.append(b'~')
        pieces.append(str(maxlen))
    if minid is not None:
        pieces.append(b'MINID')
        if approximate:
            pieces.append(b'~')
        pieces.append(minid)
    if limit is not None:
        pieces.extend([b'LIMIT', limit])
    if nomkstream:
        pieces.append(b'NOMKSTREAM')
    pieces.append(id)
    if not isinstance(fields, dict) or len(fields) == 0:
        raise DataError('XADD fields must be a non-empty dict')
    for pair in fields.items():
        pieces.extend(pair)
    return self.execute_command('XADD', name, *pieces)