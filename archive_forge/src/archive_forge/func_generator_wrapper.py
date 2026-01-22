from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
@functools.wraps(func)
def generator_wrapper(*args: Any, langsmith_extra: Optional[LangSmithExtra]=None, **kwargs: Any) -> Any:
    run_container = _setup_run(func, container_input=container_input, langsmith_extra=langsmith_extra, args=args, kwargs=kwargs)
    func_accepts_parent_run = inspect.signature(func).parameters.get('run_tree', None) is not None
    results: List[Any] = []
    try:
        if func_accepts_parent_run:
            generator_result = run_container['context'].run(func, *args, run_tree=run_container['new_run'], **kwargs)
        else:
            generator_result = run_container['context'].run(func, *args, **kwargs)
        for item in generator_result:
            if run_type == 'llm':
                if run_container['new_run']:
                    run_container['new_run'].add_event({'name': 'new_token', 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'kwargs': {'token': item}})
            results.append(item)
            try:
                yield item
            except GeneratorExit:
                break
    except BaseException as e:
        stacktrace = traceback.format_exc()
        _container_end(run_container, error=stacktrace)
        raise e
    if results:
        if reduce_fn:
            try:
                function_result = reduce_fn(results)
            except Exception as e:
                LOGGER.error(e)
                function_result = results
        else:
            function_result = results
    else:
        function_result = None
    _container_end(run_container, outputs=function_result)