import six
import sys
import time
import traceback
import random
import asyncio
import functools
def wrap_simple(f):

    @six.wraps(f)
    async def aio_wrapped_f(*args, loop=None, executor=None, **kw):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = functools.partial(Retrying().call, *args, **kw)
        return await loop.run_in_executor(executor, pfunc)
    return aio_wrapped_f