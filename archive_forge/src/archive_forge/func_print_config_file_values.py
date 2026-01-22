import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output
def print_config_file_values(self, variant: Kind) -> None:
    """Get key-value pairs from the file of a variant"""
    for name, value in self.configuration.get_values_in_config(variant).items():
        with indent_log():
            write_output('%s: %s', name, value)