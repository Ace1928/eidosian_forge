from __future__ import annotations
import asyncio
import functools
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import (
from uuid import UUID
from langsmith.run_helpers import get_run_tree_context
from tenacity import RetryCallState
from langchain_core.callbacks.base import (
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils.env import env_var_is_set
@contextmanager
def trace_as_chain_group(group_name: str, callback_manager: Optional[CallbackManager]=None, *, inputs: Optional[Dict[str, Any]]=None, project_name: Optional[str]=None, example_id: Optional[Union[str, UUID]]=None, run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None) -> Generator[CallbackManagerForChainGroup, None, None]:
    """Get a callback manager for a chain group in a context manager.
    Useful for grouping different calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name (str): The name of the chain group.
        callback_manager (CallbackManager, optional): The callback manager to use.
        inputs (Dict[str, Any], optional): The inputs to the chain group.
        project_name (str, optional): The name of the project.
            Defaults to None.
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        run_id (UUID, optional): The ID of the run.
        tags (List[str], optional): The inheritable tags to apply to all runs.
            Defaults to None.
        metadata (Dict[str, Any], optional): The metadata to apply to all runs.
            Defaults to None.

    Note: must have LANGCHAIN_TRACING_V2 env var set to true to see the trace in LangSmith.

    Returns:
        CallbackManagerForChainGroup: The callback manager for the chain group.

    Example:
        .. code-block:: python

            llm_input = "Foo"
            with trace_as_chain_group("group_name", inputs={"input": llm_input}) as manager:
                # Use the callback manager for the chain group
                res = llm.invoke(llm_input, {"callbacks": manager})
                manager.on_chain_end({"output": res})
    """
    from langchain_core.tracers.context import _get_trace_callbacks
    cb = _get_trace_callbacks(project_name, example_id, callback_manager=callback_manager)
    cm = CallbackManager.configure(inheritable_callbacks=cb, inheritable_tags=tags, inheritable_metadata=metadata)
    run_manager = cm.on_chain_start({'name': group_name}, inputs or {}, run_id=run_id)
    child_cm = run_manager.get_child()
    group_cm = CallbackManagerForChainGroup(child_cm.handlers, child_cm.inheritable_handlers, child_cm.parent_run_id, parent_run_manager=run_manager, tags=child_cm.tags, inheritable_tags=child_cm.inheritable_tags, metadata=child_cm.metadata, inheritable_metadata=child_cm.inheritable_metadata)
    try:
        yield group_cm
    except Exception as e:
        if not group_cm.ended:
            run_manager.on_chain_error(e)
        raise e
    else:
        if not group_cm.ended:
            run_manager.on_chain_end({})