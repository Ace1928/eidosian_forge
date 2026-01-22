import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@loop_asyncio.exit
def loop_asyncio_exit(kernel):
    """Exit hook for asyncio"""
    import asyncio
    loop = asyncio.get_event_loop()

    async def close_loop():
        if hasattr(loop, 'shutdown_asyncgens'):
            yield loop.shutdown_asyncgens()
        loop._should_close = True
        loop.stop()
    if loop.is_running():
        close_loop()
    elif not loop.is_closed():
        loop.run_until_complete(close_loop)
        loop.close()