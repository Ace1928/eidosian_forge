import sys
import os
import re
import pathlib
import contextlib
import logging
from email import message_from_file
from .errors import (
from .fancy_getopt import FancyGetopt, translate_longopt
from .util import check_environ, strtobool, rfc822_escape
from ._log import log
from .debug import DEBUG
def maybe_write(header, val):
    if val:
        file.write(f'{header}: {val}\n')