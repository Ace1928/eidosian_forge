import io
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import (IO, Dict, Iterable, Iterator, Mapping, Optional, Tuple,
from .parser import Binding, parse_stream
from .variables import parse_variables
def set_as_environment_variables(self) -> bool:
    """
        Load the current dotenv as system environment variable.
        """
    if not self.dict():
        return False
    for k, v in self.dict().items():
        if k in os.environ and (not self.override):
            continue
        if v is not None:
            os.environ[k] = v
    return True