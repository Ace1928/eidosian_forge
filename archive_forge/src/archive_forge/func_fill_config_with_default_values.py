import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
def fill_config_with_default_values(config: ConfigParser, default_values: Mapping[str, Mapping[str, Any]]) -> None:
    for section in default_values.keys():
        if not config.has_section(section):
            config.add_section(section)
        for opt, val in default_values[section].items():
            if not config.has_option(section, opt):
                config.set(section, opt, str(val))