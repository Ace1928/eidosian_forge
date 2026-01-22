from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
def remove_timeout_callback(self, callback_id: ID) -> None:
    """ Removes a callback added with add_timeout_callback, before it runs.

        """
    self._execute_remover(callback_id, self._timeout_callback_removers)