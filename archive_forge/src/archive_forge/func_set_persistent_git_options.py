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
def set_persistent_git_options(self, **kwargs: Any) -> None:
    """Specify command line options to the git executable for subsequent
        subcommand calls.

        :param kwargs:
            A dict of keyword arguments.
            These arguments are passed as in :meth:`_call_process`, but will be
            passed to the git command rather than the subcommand.
        """
    self._persistent_git_options = self.transform_kwargs(split_single_char_options=True, **kwargs)