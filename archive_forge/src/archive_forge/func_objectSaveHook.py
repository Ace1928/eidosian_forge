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
def objectSaveHook(pythonObject: object) -> JSONDict:
    """
    Object-to-serializable hook for certain value types used within the logging
    system.

    @see: the C{default} parameter to L{json.dump}

    @param pythonObject: Any object.

    @return: If the object is one of the special types the logging system
        supports, a specially-formatted dictionary; otherwise, a marker
        dictionary indicating that it could not be serialized.
    """
    for predicate, uuid, saver, loader in classInfo:
        if predicate(pythonObject):
            result = saver(pythonObject)
            result['__class_uuid__'] = str(uuid)
            return result
    return {'unpersistable': True}