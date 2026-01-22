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
def objectLoadHook(aDict: JSONDict) -> object:
    """
    Dictionary-to-object-translation hook for certain value types used within
    the logging system.

    @see: the C{object_hook} parameter to L{json.load}

    @param aDict: A dictionary loaded from a JSON object.

    @return: C{aDict} itself, or the object represented by C{aDict}
    """
    if '__class_uuid__' in aDict:
        return uuidToLoader[UUID(aDict['__class_uuid__'])](aDict)
    return aDict