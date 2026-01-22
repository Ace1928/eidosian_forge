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
def get_bedrock_anthropic_callback() -> Generator[BedrockAnthropicTokenUsageCallbackHandler, None, None]:
    """Get the Bedrock anthropic callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        BedrockAnthropicTokenUsageCallbackHandler:
            The Bedrock anthropic callback handler.

    Example:
        >>> with get_bedrock_anthropic_callback() as cb:
        ...     # Use the Bedrock anthropic callback handler
    """
    cb = BedrockAnthropicTokenUsageCallbackHandler()
    bedrock_anthropic_callback_var.set(cb)
    yield cb
    bedrock_anthropic_callback_var.set(None)