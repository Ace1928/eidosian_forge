from __future__ import annotations
import sys
from http.client import responses
from typing import TYPE_CHECKING
from vine import Thenable, maybe_promise, promise
from kombu.exceptions import HttpError
from kombu.utils.compat import coro
from kombu.utils.encoding import bytes_to_str
from kombu.utils.functional import maybe_list, memoize
def raise_for_error(self):
    """Raise if the request resulted in an HTTP error code.

        Raises
        ------
            :class:`~kombu.exceptions.HttpError`
        """
    if self.error:
        raise self.error