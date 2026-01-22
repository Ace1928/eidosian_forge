import asyncio
import contextlib
import warnings
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Type, Union
import pytest
from aiohttp.helpers import isasyncgenfunction
from aiohttp.web import Application
from .test_utils import (
@pytest.fixture
def loop_debug(request):
    """--enable-loop-debug config option"""
    return request.config.getoption('--aiohttp-enable-loop-debug')