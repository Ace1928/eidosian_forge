from __future__ import annotations
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from langchain_core.tracers.context import register_configure_hook
from langchain_community.callbacks.bedrock_anthropic_callback import (
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.callbacks.tracers.comet import CometTracer
from langchain_community.callbacks.tracers.wandb import WandbTracer
@contextmanager
def wandb_tracing_enabled(session_name: str='default') -> Generator[None, None, None]:
    """Get the WandbTracer in a context manager.

    Args:
        session_name (str, optional): The name of the session.
            Defaults to "default".

    Returns:
        None

    Example:
        >>> with wandb_tracing_enabled() as session:
        ...     # Use the WandbTracer session
    """
    cb = WandbTracer()
    wandb_tracing_callback_var.set(cb)
    yield None
    wandb_tracing_callback_var.set(None)