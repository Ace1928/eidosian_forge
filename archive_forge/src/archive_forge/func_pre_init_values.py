import re
import traceback
import types
from collections import OrderedDict
from os import getenv, path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, NamedTuple,
from sphinx.errors import ConfigError, ExtensionError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.i18n import format_date
from sphinx.util.osutil import cd, fs_encoding
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
def pre_init_values(self) -> None:
    """
        Initialize some limited config variables before initializing i18n and loading
        extensions.
        """
    variables = ['needs_sphinx', 'suppress_warnings', 'language', 'locale_dirs']
    for name in variables:
        try:
            if name in self.overrides:
                self.__dict__[name] = self.convert_overrides(name, self.overrides[name])
            elif name in self._raw_config:
                self.__dict__[name] = self._raw_config[name]
        except ValueError as exc:
            logger.warning('%s', exc)