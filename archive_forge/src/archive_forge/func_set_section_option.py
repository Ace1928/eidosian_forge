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
def set_section_option(self, section: str, name: str, value: str) -> None:
    """Set an option programmatically within the given section.

        The section is created if it doesn't exist already.
        The value here will override whatever was in the .ini
        file.

        :param section: name of the section

        :param name: name of the value

        :param value: the value.  Note that this value is passed to
         ``ConfigParser.set``, which supports variable interpolation using
         pyformat (e.g. ``%(some_value)s``).   A raw percent sign not part of
         an interpolation symbol must therefore be escaped, e.g. ``%%``.
         The given value may refer to another value already in the file
         using the interpolation format.

        """
    if not self.file_config.has_section(section):
        self.file_config.add_section(section)
    self.file_config.set(section, name, value)