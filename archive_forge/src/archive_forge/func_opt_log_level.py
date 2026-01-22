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
def opt_log_level(self, levelName: str) -> None:
    """
        Set default log level.
        (options: {options}; default: "{default}")
        """
    try:
        self['logLevel'] = LogLevel.levelWithName(levelName)
    except InvalidLogLevelError:
        raise UsageError(f'Invalid log level: {levelName}')