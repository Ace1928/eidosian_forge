from __future__ import annotations
import functools
import linecache
import logging
from typing import Match
from typing import NamedTuple
from flake8 import defaults
from flake8 import utils
Determine if a comment has been added to ignore this line.

        :param disable_noqa:
            Whether or not users have provided ``--disable-noqa``.
        :returns:
            True if error is ignored in-line, False otherwise.
        