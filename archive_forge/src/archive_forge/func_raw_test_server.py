import asyncio
import contextlib
import warnings
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Type, Union
import pytest
from aiohttp.helpers import isasyncgenfunction
from aiohttp.web import Application
from .test_utils import (
@pytest.fixture
def raw_test_server(aiohttp_raw_server):
    warnings.warn('Deprecated, use aiohttp_raw_server fixture instead', DeprecationWarning, stacklevel=2)
    return aiohttp_raw_server