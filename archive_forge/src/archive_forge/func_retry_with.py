import functools
import sys
import threading
import time
import typing as t
import warnings
from abc import ABC, abstractmethod
from concurrent import futures
from inspect import iscoroutinefunction
from .retry import retry_base  # noqa
from .retry import retry_all  # noqa
from .retry import retry_always  # noqa
from .retry import retry_any  # noqa
from .retry import retry_if_exception  # noqa
from .retry import retry_if_exception_type  # noqa
from .retry import retry_if_exception_cause_type  # noqa
from .retry import retry_if_not_exception_type  # noqa
from .retry import retry_if_not_result  # noqa
from .retry import retry_if_result  # noqa
from .retry import retry_never  # noqa
from .retry import retry_unless_exception_type  # noqa
from .retry import retry_if_exception_message  # noqa
from .retry import retry_if_not_exception_message  # noqa
from .nap import sleep  # noqa
from .nap import sleep_using_event  # noqa
from .stop import stop_after_attempt  # noqa
from .stop import stop_after_delay  # noqa
from .stop import stop_all  # noqa
from .stop import stop_any  # noqa
from .stop import stop_never  # noqa
from .stop import stop_when_event_set  # noqa
from .wait import wait_chain  # noqa
from .wait import wait_combine  # noqa
from .wait import wait_exponential  # noqa
from .wait import wait_fixed  # noqa
from .wait import wait_incrementing  # noqa
from .wait import wait_none  # noqa
from .wait import wait_random  # noqa
from .wait import wait_random_exponential  # noqa
from .wait import wait_random_exponential as wait_full_jitter  # noqa
from .wait import wait_exponential_jitter  # noqa
from .before import before_log  # noqa
from .before import before_nothing  # noqa
from .after import after_log  # noqa
from .after import after_nothing  # noqa
from .before_sleep import before_sleep_log  # noqa
from .before_sleep import before_sleep_nothing  # noqa
from pip._vendor.tenacity._asyncio import AsyncRetrying  # noqa:E402,I100
def retry_with(*args: t.Any, **kwargs: t.Any) -> WrappedFn:
    return self.copy(*args, **kwargs).wraps(f)