from __future__ import annotations
import argparse
import contextlib
import copy
import enum
import functools
import logging
from typing import Generator
from typing import Sequence
from flake8 import defaults
from flake8 import statistics
from flake8 import utils
from flake8.formatting import base as base_formatter
from flake8.violation import Violation
def was_ignored(self, code: str) -> Selected | Ignored:
    """Determine if the code has been ignored by the user.

        :param code:
            The code for the check that has been run.
        :returns:
            Selected.Implicitly if the ignored list is empty,
            Ignored.Explicitly if the ignored list is not empty and a match was
            found,
            Selected.Implicitly if the ignored list is not empty but no match
            was found.
        """
    if code.startswith(self.ignored_explicitly):
        return Ignored.Explicitly
    elif code.startswith(self.ignored):
        return Ignored.Implicitly
    else:
        return Selected.Implicitly