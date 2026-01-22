import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zadd(self, name: KeyT, mapping: Mapping[AnyKeyT, EncodableT], nx: bool=False, xx: bool=False, ch: bool=False, incr: bool=False, gt: bool=False, lt: bool=False) -> ResponseT:
    """
        Set any number of element-name, score pairs to the key ``name``. Pairs
        are specified as a dict of element-names keys to score values.

        ``nx`` forces ZADD to only create new elements and not to update
        scores for elements that already exist.

        ``xx`` forces ZADD to only update scores of elements that already
        exist. New elements will not be added.

        ``ch`` modifies the return value to be the numbers of elements changed.
        Changed elements include new elements that were added and elements
        whose scores changed.

        ``incr`` modifies ZADD to behave like ZINCRBY. In this mode only a
        single element/score pair can be specified and the score is the amount
        the existing score will be incremented by. When using this mode the
        return value of ZADD will be the new score of the element.

        ``LT`` Only update existing elements if the new score is less than
        the current score. This flag doesn't prevent adding new elements.

        ``GT`` Only update existing elements if the new score is greater than
        the current score. This flag doesn't prevent adding new elements.

        The return value of ZADD varies based on the mode specified. With no
        options, ZADD returns the number of new elements added to the sorted
        set.

        ``NX``, ``LT``, and ``GT`` are mutually exclusive options.

        See: https://redis.io/commands/ZADD
        """
    if not mapping:
        raise DataError('ZADD requires at least one element/score pair')
    if nx and xx:
        raise DataError("ZADD allows either 'nx' or 'xx', not both")
    if gt and lt:
        raise DataError("ZADD allows either 'gt' or 'lt', not both")
    if incr and len(mapping) != 1:
        raise DataError("ZADD option 'incr' only works when passing a single element/score pair")
    if nx and (gt or lt):
        raise DataError("Only one of 'nx', 'lt', or 'gr' may be defined.")
    pieces: list[EncodableT] = []
    options = {}
    if nx:
        pieces.append(b'NX')
    if xx:
        pieces.append(b'XX')
    if ch:
        pieces.append(b'CH')
    if incr:
        pieces.append(b'INCR')
        options['as_score'] = True
    if gt:
        pieces.append(b'GT')
    if lt:
        pieces.append(b'LT')
    for pair in mapping.items():
        pieces.append(pair[1])
        pieces.append(pair[0])
    return self.execute_command('ZADD', name, *pieces, **options)