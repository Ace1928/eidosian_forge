import io
from typing import IO, Any, List, Optional, TextIO, Tuple, Type, cast
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._file import textFileLogObserver
from .._global import MORE_THAN_ONCE_WARNING, LogBeginner
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
from ..test.test_stdlib import nextLine

                Emulate warnings.showwarning.

                @param message: A warning message to emit.
                @param category: A warning category to associate with
                    C{message}.
                @param filename: A file name for the source code file issuing
                    the warning.
                @param lineno: A line number in the source file where the
                    warning was issued.
                @param file: A file to write the warning message to.  If
                    L{None}, write to L{sys.stderr}.
                @param line: A line of source code to include with the warning
                    message. If L{None}, attempt to read the line from
                    C{filename} and C{lineno}.
                