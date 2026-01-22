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
def request_expiration(self) -> None:
    """ Used in test suite for now. Forces immediate expiration if no connections."""
    self._expiration_requested = True