from json import dumps, loads
from typing import IO, Any, AnyStr, Dict, Iterable, Optional, Union, cast
from uuid import UUID
from constantly import NamedConstant
from twisted.python.failure import Failure
from ._file import FileLogObserver
from ._flatten import flattenEvent
from ._interfaces import LogEvent
from ._levels import LogLevel
from ._logger import Logger
def eventFromRecord(record: bytearray) -> Optional[LogEvent]:
    if record[-1] == ord('\n'):
        return eventFromBytearray(record)
    else:
        log.error('Unable to read truncated JSON record: {record!r}', record=bytes(record))
    return None