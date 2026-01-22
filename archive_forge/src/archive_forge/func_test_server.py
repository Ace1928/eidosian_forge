import asyncio
import contextlib
import warnings
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Type, Union
import pytest
from aiohttp.helpers import isasyncgenfunction
from aiohttp.web import Application
from .test_utils import (
@pytest.fixture
def test_server(aiohttp_server):
    warnings.warn('Deprecated, use aiohttp_server fixture instead', DeprecationWarning, stacklevel=2)
    return aiohttp_server