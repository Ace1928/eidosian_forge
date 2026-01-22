from __future__ import annotations
import asyncio
import uuid
import warnings
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
from typing import (
from typing_extensions import ParamSpec, TypedDict
from langchain_core.runnables.utils import (
def call_func_with_variable_args(func: Union[Callable[[Input], Output], Callable[[Input, RunnableConfig], Output], Callable[[Input, CallbackManagerForChainRun], Output], Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output]], input: Input, config: RunnableConfig, run_manager: Optional[CallbackManagerForChainRun]=None, **kwargs: Any) -> Output:
    """Call function that may optionally accept a run_manager and/or config.

    Args:
        func (Union[Callable[[Input], Output],
          Callable[[Input, CallbackManagerForChainRun], Output],
          Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output]]):
           The function to call.
        input (Input): The input to the function.
        run_manager (CallbackManagerForChainRun): The run manager to
          pass to the function.
        config (RunnableConfig): The config to pass to the function.
        **kwargs (Any): The keyword arguments to pass to the function.

    Returns:
        Output: The output of the function.
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs['config'] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs['config'] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs['run_manager'] = run_manager
    return func(input, **kwargs)