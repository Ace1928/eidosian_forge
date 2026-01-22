import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional
import numpy as np
from numpy.lib.function_base import (
def vectorize_call_coroutine(self, broadcast_shape, args, kwargs):
    """Run coroutines concurrently.

        Creates as many tasks as needed and executes them in a new event
        loop.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """

    async def create_and_gather_tasks():
        tasks = []
        for index in np.ndindex(*broadcast_shape):
            current_args = tuple((arg[index] for arg in args))
            current_kwargs = {key: value[index] for key, value in kwargs.items()}
            tasks.append(self.func(*current_args, **current_kwargs))
        outputs = await asyncio.gather(*tasks)
        return outputs
    loop = asyncio.new_event_loop()
    try:
        outputs = loop.run_until_complete(create_and_gather_tasks())
    finally:
        loop.close()
    return outputs