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
def convert_overrides(self, name: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    else:
        defvalue = self.values[name][0]
        if self.values[name][2] == Any:
            return value
        elif self.values[name][2] == {bool, str}:
            if value == '0':
                return False
            elif value == '1':
                return True
            else:
                return value
        elif type(defvalue) is bool or self.values[name][2] == [bool]:
            if value == '0':
                return False
            else:
                return bool(value)
        elif isinstance(defvalue, dict):
            raise ValueError(__('cannot override dictionary config setting %r, ignoring (use %r to set individual elements)') % (name, name + '.key=value'))
        elif isinstance(defvalue, list):
            return value.split(',')
        elif isinstance(defvalue, int):
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(__('invalid number %r for config value %r, ignoring') % (value, name)) from exc
        elif callable(defvalue):
            return value
        elif defvalue is not None and (not isinstance(defvalue, str)):
            raise ValueError(__('cannot override config setting %r with unsupported type, ignoring') % name)
        else:
            return value