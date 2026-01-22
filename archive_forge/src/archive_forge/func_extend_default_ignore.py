from __future__ import annotations
import argparse
import enum
import functools
import logging
from typing import Any
from typing import Callable
from typing import Sequence
from flake8 import utils
from flake8.plugins.finder import Plugins
def extend_default_ignore(self, error_codes: Sequence[str]) -> None:
    """Extend the default ignore list with the error codes provided.

        :param error_codes:
            List of strings that are the error/warning codes with which to
            extend the default ignore list.
        """
    LOG.debug('Extending default ignore list with %r', error_codes)
    self.extended_default_ignore.extend(error_codes)