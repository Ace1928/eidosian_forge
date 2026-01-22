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
def merge_configs(*configs: Optional[RunnableConfig]) -> RunnableConfig:
    """Merge multiple configs into one.

    Args:
        *configs (Optional[RunnableConfig]): The configs to merge.

    Returns:
        RunnableConfig: The merged config.
    """
    base: RunnableConfig = {}
    for config in (c for c in configs if c is not None):
        for key in config:
            if key == 'metadata':
                base[key] = {**base.get(key, {}), **(config.get(key) or {})}
            elif key == 'tags':
                base[key] = list(set(base.get(key, []) + (config.get(key) or [])))
            elif key == 'configurable':
                base[key] = {**base.get(key, {}), **(config.get(key) or {})}
            elif key == 'callbacks':
                base_callbacks = base.get('callbacks')
                these_callbacks = config['callbacks']
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base['callbacks'] = these_callbacks
                    elif isinstance(base_callbacks, list):
                        base['callbacks'] = base_callbacks + these_callbacks
                    else:
                        mngr = base_callbacks.copy()
                        for callback in these_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base['callbacks'] = mngr
                elif these_callbacks is not None:
                    if base_callbacks is None:
                        base['callbacks'] = these_callbacks
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base['callbacks'] = mngr
                    else:
                        base['callbacks'] = base_callbacks.__class__(parent_run_id=base_callbacks.parent_run_id or these_callbacks.parent_run_id, handlers=base_callbacks.handlers + these_callbacks.handlers, inheritable_handlers=base_callbacks.inheritable_handlers + these_callbacks.inheritable_handlers, tags=list(set(base_callbacks.tags + these_callbacks.tags)), inheritable_tags=list(set(base_callbacks.inheritable_tags + these_callbacks.inheritable_tags)), metadata={**base_callbacks.metadata, **these_callbacks.metadata})
            else:
                base[key] = config[key] or base.get(key)
    return base