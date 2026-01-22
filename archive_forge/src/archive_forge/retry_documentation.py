import abc
import asyncio
import datetime
import functools
import logging
import os
import random
import threading
import time
from typing import Any, Awaitable, Callable, Generic, Optional, Tuple, Type, TypeVar
from requests import HTTPError
import wandb
from wandb.util import CheckRetryFnType
from .mailbox import ContextCancelledError
Call the wrapped function, with retries.

        Arguments:
           retry_timedelta (kwarg): amount of time to retry before giving up.
           sleep_base (kwarg): amount of time to sleep upon first failure, all other sleeps
               are derived from this one.
        