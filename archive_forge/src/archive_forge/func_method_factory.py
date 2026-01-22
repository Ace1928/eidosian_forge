import asyncio
import functools
import inspect
import logging
import sys
from typing import Any, Dict, Optional, Sequence, TypeVar
import wandb.sdk
import wandb.util
from wandb.sdk.lib import telemetry as wb_telemetry
from wandb.sdk.lib.timer import Timer
def method_factory(original_method: Any):

    async def async_method(*args, **kwargs):
        future = asyncio.Future()

        async def callback(coro):
            try:
                result = await coro
                loggable_dict = self.resolver(args, kwargs, result, timer.start_time, timer.elapsed)
                if loggable_dict is not None:
                    run.log(loggable_dict)
                future.set_result(result)
            except Exception as e:
                logger.warning(e)
        with Timer() as timer:
            coro = original_method(*args, **kwargs)
            asyncio.ensure_future(callback(coro))
        return await future

    def sync_method(*args, **kwargs):
        with Timer() as timer:
            result = original_method(*args, **kwargs)
            try:
                loggable_dict = self.resolver(args, kwargs, result, timer.start_time, timer.elapsed)
                if loggable_dict is not None:
                    run.log(loggable_dict)
            except Exception as e:
                logger.warning(e)
            return result
    if inspect.iscoroutinefunction(original_method):
        return functools.wraps(original_method)(async_method)
    else:
        return functools.wraps(original_method)(sync_method)