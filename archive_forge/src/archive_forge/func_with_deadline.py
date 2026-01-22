from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def with_deadline(self, deadline: float | None) -> Self:
    """Return a copy of this retry with the given timeout.

        DEPRECATED: use :meth:`with_timeout` instead. Refer to the ``Retry`` class
        documentation for details.

        Args:
            deadline (float|None): How long to keep retrying, in seconds. If None,
                no timeout is enforced.

        Returns:
            Retry: A new retry instance with the given timeout.
        """
    return self.with_timeout(deadline)