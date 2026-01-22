import types
import warnings
from typing import Any, Awaitable, Callable, Dict, Tuple
from .abc import AbstractStreamWriter
from .payload import Payload, payload_type

Payload implementation for coroutines as data provider.

As a simple case, you can upload data from file::

   @aiohttp.streamer
   async def file_sender(writer, file_name=None):
      with open(file_name, 'rb') as f:
          chunk = f.read(2**16)
          while chunk:
              await writer.write(chunk)

              chunk = f.read(2**16)

Then you can use `file_sender` like this:

    async with session.post('http://httpbin.org/post',
                            data=file_sender(file_name='huge_file')) as resp:
        print(await resp.text())

..note:: Coroutine must accept `writer` as first argument

