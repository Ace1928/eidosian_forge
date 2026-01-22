import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def event_loop_thread_exec(func: Any) -> Any:
    """Wrapper for running any function in an awaitable thread on an event loop.

    Example usage:
    ```
    def my_func(arg1, arg2):
        return arg1 + arg2

    future = event_loop_thread_exec(my_func)(2, 2)
    assert await future == 4
    ```

    The returned function must be called within an active event loop.
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        result = cast(Any, await loop.run_in_executor(None, lambda: func(*args, **kwargs)))
        return result
    return wrapper