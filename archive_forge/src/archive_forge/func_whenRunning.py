from os import kill
from signal import SIGTERM
from sys import stderr
from typing import Any, Callable, Mapping, TextIO
from attr import Factory, attrib, attrs
from constantly import NamedConstant
from twisted.internet.interfaces import IReactorCore
from twisted.logger import (
from ._exit import ExitStatus, exit
from ._pidfile import AlreadyRunningError, InvalidPIDFileError, IPIDFile, nonePIDFile
def whenRunning(self) -> None:
    """
        Call C{self._whenRunning} with C{self._whenRunningArguments}.

        @note: This method is called after the reactor starts running.
        """
    self._whenRunning(**self._whenRunningArguments)