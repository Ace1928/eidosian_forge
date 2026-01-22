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