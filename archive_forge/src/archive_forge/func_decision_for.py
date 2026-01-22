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
def decision_for(self, code: str) -> Decision:
    """Return the decision for a specific code.

        This method caches the decisions for codes to avoid retracing the same
        logic over and over again. We only care about the select and ignore
        rules as specified by the user in their configuration files and
        command-line flags.

        This method does not look at whether the specific line is being
        ignored in the file itself.

        :param code: The code for the check that has been run.
        """
    decision = self.cache.get(code)
    if decision is None:
        decision = self.make_decision(code)
        self.cache[code] = decision
        LOG.debug('"%s" will be "%s"', code, decision)
    return decision