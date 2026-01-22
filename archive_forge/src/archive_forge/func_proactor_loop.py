import asyncio
import contextlib
import warnings
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Type, Union
import pytest
from aiohttp.helpers import isasyncgenfunction
from aiohttp.web import Application
from .test_utils import (
@pytest.fixture
def proactor_loop():
    policy = asyncio.WindowsProactorEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)
    with loop_context(policy.new_event_loop) as _loop:
        asyncio.set_event_loop(_loop)
        yield _loop