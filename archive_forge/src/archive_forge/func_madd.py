from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def madd(self, ktv_tuples: List[Tuple[KeyT, Union[int, str], Number]]):
    """
        Append (or create and append) a new `value` to series
        `key` with `timestamp`.
        Expects a list of `tuples` as (`key`,`timestamp`, `value`).
        Return value is an array with timestamps of insertions.

        For more information: https://redis.io/commands/ts.madd/
        """
    params = []
    for ktv in ktv_tuples:
        params.extend(ktv)
    return self.execute_command(MADD_CMD, *params)