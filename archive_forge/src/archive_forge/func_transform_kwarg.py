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
def transform_kwarg(self, name: str, value: Any, split_single_char_options: bool) -> List[str]:
    if len(name) == 1:
        if value is True:
            return ['-%s' % name]
        elif value not in (False, None):
            if split_single_char_options:
                return ['-%s' % name, '%s' % value]
            else:
                return ['-%s%s' % (name, value)]
    elif value is True:
        return ['--%s' % dashify(name)]
    elif value is not False and value is not None:
        return ['--%s=%s' % (dashify(name), value)]
    return []