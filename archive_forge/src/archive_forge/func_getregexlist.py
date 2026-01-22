from __future__ import annotations
import collections
import configparser
import copy
import os
import os.path
import re
from typing import (
from coverage.exceptions import ConfigError
from coverage.misc import isolate_module, human_sorted_items, substitute_variables
from coverage.tomlconfig import TomlConfigParser, TomlDecodeError
from coverage.types import (
def getregexlist(self, section: str, option: str) -> list[str]:
    """Read a list of full-line regexes.

        The value of `section` and `option` is treated as a newline-separated
        list of regexes.  Each value is stripped of white space.

        Returns the list of strings.

        """
    line_list = self.get(section, option)
    value_list = []
    for value in line_list.splitlines():
        value = value.strip()
        try:
            re.compile(value)
        except re.error as e:
            raise ConfigError(f'Invalid [{section}].{option} value {value!r}: {e}') from e
        if value:
            value_list.append(value)
    return value_list