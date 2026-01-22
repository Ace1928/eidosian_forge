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
def failureFromJSON(failureDict: JSONDict) -> Failure:
    """
    Load a L{Failure} from a dictionary deserialized from JSON.

    @param failureDict: a JSON-deserialized object like one previously returned
        by L{failureAsJSON}.

    @return: L{Failure}
    """
    f = Failure.__new__(Failure)
    typeInfo = failureDict['type']
    failureDict['type'] = type(typeInfo['__name__'], (), typeInfo)
    f.__dict__ = failureDict
    return f