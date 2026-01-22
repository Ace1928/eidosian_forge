import typing
from sys import stderr, stdout
from textwrap import dedent
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, cast
from twisted.copyright import version
from twisted.internet.interfaces import IReactorCore
from twisted.logger import (
from twisted.plugin import getPlugins
from twisted.python.usage import Options, UsageError
from ..reactors import NoSuchReactor, getReactorTypes, installReactor
from ..runner._exit import ExitStatus, exit
from ..service import IServiceMaker
def selectDefaultLogObserver(self) -> None:
    """
        Set C{fileLogObserverFactory} to the default appropriate for the
        chosen C{logFile}.
        """
    if 'fileLogObserverFactory' not in self:
        logFile = self['logFile']
        if hasattr(logFile, 'isatty') and logFile.isatty():
            self['fileLogObserverFactory'] = textFileLogObserver
            self['logFormat'] = 'text'
        else:
            self['fileLogObserverFactory'] = jsonFileLogObserver
            self['logFormat'] = 'json'