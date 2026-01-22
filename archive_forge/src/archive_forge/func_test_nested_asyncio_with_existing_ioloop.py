import asyncio
from unittest.mock import MagicMock
import pytest
import tornado
from nbclient.util import run_hook, run_sync
def test_nested_asyncio_with_existing_ioloop():

    async def _test():
        assert some_async_function() == 42
        return asyncio.get_running_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    event_loop = loop.run_until_complete(_test())
    assert event_loop is loop