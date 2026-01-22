from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
from configparser import ConfigParser
import inspect
import os
import sys
from typing import Any
from typing import cast
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Union
from typing_extensions import TypedDict
from . import __version__
from . import command
from . import util
from .util import compat
def get_section_option(self, section: str, name: str, default: Optional[str]=None) -> Optional[str]:
    """Return an option from the given section of the .ini file."""
    if not self.file_config.has_section(section):
        raise util.CommandError("No config file %r found, or file has no '[%s]' section" % (self.config_file_name, section))
    if self.file_config.has_option(section, name):
        return self.file_config.get(section, name)
    else:
        return default