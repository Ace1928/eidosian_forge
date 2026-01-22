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
def eventAsJSON(event: LogEvent) -> str:
    """
    Encode an event as JSON, flattening it if necessary to preserve as much
    structure as possible.

    Not all structure from the log event will be preserved when it is
    serialized.

    @param event: A log event dictionary.

    @return: A string of the serialized JSON; note that this will contain no
        newline characters, and may thus safely be stored in a line-delimited
        file.
    """

    def default(unencodable: object) -> Union[JSONDict, str]:
        """
        Serialize an object not otherwise serializable by L{dumps}.

        @param unencodable: An unencodable object.

        @return: C{unencodable}, serialized
        """
        if isinstance(unencodable, bytes):
            return unencodable.decode('charmap')
        return objectSaveHook(unencodable)
    flattenEvent(event)
    return dumps(event, default=default, skipkeys=True)