from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def process_scoped_temporary_file(args: CommonConfig, prefix: t.Optional[str]='ansible-test-', suffix: t.Optional[str]=None) -> str:
    """Return the path to a temporary file that will be automatically removed when the process exits."""
    if args.explain:
        path = os.path.join(tempfile.gettempdir(), f'{prefix or tempfile.gettempprefix()}{generate_name()}{suffix or ''}')
    else:
        temp_fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(temp_fd)
        ExitHandler.register(lambda: os.remove(path))
    return path