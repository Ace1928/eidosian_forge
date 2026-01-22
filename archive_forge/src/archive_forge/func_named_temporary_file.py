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
@contextlib.contextmanager
def named_temporary_file(args: CommonConfig, prefix: str, suffix: str, directory: t.Optional[str], content: str) -> c.Iterator[str]:
    """Context manager for a named temporary file."""
    if args.explain:
        yield os.path.join(directory or '/tmp', '%stemp%s' % (prefix, suffix))
    else:
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, dir=directory) as tempfile_fd:
            tempfile_fd.write(to_bytes(content))
            tempfile_fd.flush()
            yield tempfile_fd.name