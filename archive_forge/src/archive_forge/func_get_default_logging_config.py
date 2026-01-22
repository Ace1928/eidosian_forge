from __future__ import annotations
import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
from traitlets.traitlets import (
from traitlets.utils.bunch import Bunch
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs
from ..utils import cast_unicode
from ..utils.importstring import import_item
def get_default_logging_config(self) -> StrDict:
    """Return the base logging configuration.

        The default is to log to stderr using a StreamHandler, if no default
        handler already exists.

        The log handler level starts at logging.WARN, but this can be adjusted
        by setting the ``log_level`` attribute.

        The ``logging_config`` trait is merged into this allowing for finer
        control of logging.

        """
    config: StrDict = {'version': 1, 'handlers': {'console': {'class': 'logging.StreamHandler', 'formatter': 'console', 'level': logging.getLevelName(self.log_level), 'stream': 'ext://sys.stderr'}}, 'formatters': {'console': {'class': f'{self._log_formatter_cls.__module__}.{self._log_formatter_cls.__name__}', 'format': self.log_format, 'datefmt': self.log_datefmt}}, 'loggers': {self.__class__.__name__: {'level': 'DEBUG', 'handlers': ['console']}}, 'disable_existing_loggers': False}
    if IS_PYTHONW:
        del config['handlers']
        del config['loggers']
    return config