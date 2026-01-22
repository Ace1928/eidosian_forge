import asyncio
from unittest.mock import MagicMock
import pytest
import tornado
from nbclient.util import run_hook, run_sync
def some_sync_function():
    return run_sync(some_async_function)()