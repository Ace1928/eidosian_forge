import configparser
import locale
import os
import sys
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple
from pip._internal.exceptions import (
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum
def unset_value(self, key: str) -> None:
    """Unset a value in the configuration."""
    orig_key = key
    key = _normalize_name(key)
    self._ensure_have_load_only()
    assert self.load_only
    if key not in self._config[self.load_only]:
        raise ConfigurationError(f'No such key - {orig_key}')
    fname, parser = self._get_parser_to_modify()
    if parser is not None:
        section, name = _disassemble_key(key)
        if not (parser.has_section(section) and parser.remove_option(section, name)):
            raise ConfigurationError('Fatal Internal error [id=1]. Please report as a bug.')
        if not parser.items(section):
            parser.remove_section(section)
        self._mark_as_modified(fname, parser)
    del self._config[self.load_only][key]