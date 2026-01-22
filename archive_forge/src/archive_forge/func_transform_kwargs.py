from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
def transform_kwargs(self, split_single_char_options: bool=True, **kwargs: Any) -> List[str]:
    """Transform Python style kwargs into git command line options."""
    args = []
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple)):
            for value in v:
                args += self.transform_kwarg(k, value, split_single_char_options)
        else:
            args += self.transform_kwarg(k, v, split_single_char_options)
    return args