from __future__ import annotations
import logging # isort:skip
import inspect
import time
from copy import copy
from functools import wraps
from typing import (
from tornado import locks
from ..events import ConnectionLost
from ..util.token import generate_jwt_token
from .callbacks import DocumentCallbackGroup
def notify_connection_lost(self) -> None:
    """ Notify the document that the connection was lost. """
    self.document.callbacks.trigger_event(ConnectionLost())