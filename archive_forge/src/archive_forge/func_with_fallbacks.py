from __future__ import annotations
import asyncio
import collections
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
from contextvars import copy_context
from functools import wraps
from itertools import groupby, tee
from operator import itemgetter
from typing import (
from typing_extensions import Literal, get_args
from langchain_core._api import beta_decorator
from langchain_core.load.dump import dumpd
from langchain_core.load.serializable import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
def with_fallbacks(self, fallbacks: Sequence[Runnable[Input, Output]], *, exceptions_to_handle: Tuple[Type[BaseException], ...]=(Exception,), exception_key: Optional[str]=None) -> RunnableWithFallbacksT[Input, Output]:
    """Add fallbacks to a runnable, returning a new Runnable.

        Example:

            .. code-block:: python

                from typing import Iterator

                from langchain_core.runnables import RunnableGenerator


                def _generate_immediate_error(input: Iterator) -> Iterator[str]:
                    raise ValueError()
                    yield ""


                def _generate(input: Iterator) -> Iterator[str]:
                    yield from "foo bar"


                runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
                    [RunnableGenerator(_generate)]
                    )
                print(''.join(runnable.stream({}))) #foo bar

        Args:
            fallbacks: A sequence of runnables to try if the original runnable fails.
            exceptions_to_handle: A tuple of exception types to handle.
            exception_key: If string is specified then handled exceptions will be passed
                to fallbacks as part of the input under the specified key. If None,
                exceptions will not be passed to fallbacks. If used, the base runnable
                and its fallbacks must accept a dictionary as input.

        Returns:
            A new Runnable that will try the original runnable, and then each
            fallback in order, upon failures.

        """
    from langchain_core.runnables.fallbacks import RunnableWithFallbacks
    return RunnableWithFallbacks(runnable=self, fallbacks=fallbacks, exceptions_to_handle=exceptions_to_handle, exception_key=exception_key)