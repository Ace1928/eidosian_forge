from __future__ import annotations
import types
from abc import ABCMeta, abstractmethod
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable
from typing import Any, TypeVar
@abstractmethod
def run_fixture(self, fixture_func: Callable[..., Coroutine[Any, Any, _T]], kwargs: dict[str, Any]) -> _T:
    """
        Run an async fixture.

        :param fixture_func: the fixture function
        :param kwargs: keyword arguments to call the fixture function with
        :return: the return value of the fixture function
        """