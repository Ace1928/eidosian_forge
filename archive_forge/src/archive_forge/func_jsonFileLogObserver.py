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
def jsonFileLogObserver(outFile: IO[Any], recordSeparator: str='\x1e') -> FileLogObserver:
    """
    Create a L{FileLogObserver} that emits JSON-serialized events to a
    specified (writable) file-like object.

    Events are written in the following form::

        RS + JSON + NL

    C{JSON} is the serialized event, which is JSON text.  C{NL} is a newline
    (C{"\\n"}).  C{RS} is a record separator.  By default, this is a single
    RS character (C{"\\x1e"}), which makes the default output conform to the
    IETF draft document "draft-ietf-json-text-sequence-13".

    @param outFile: A file-like object.  Ideally one should be passed which
        accepts L{str} data.  Otherwise, UTF-8 L{bytes} will be used.
    @param recordSeparator: The record separator to use.

    @return: A file log observer.
    """
    return FileLogObserver(outFile, lambda event: f'{recordSeparator}{eventAsJSON(event)}\n')