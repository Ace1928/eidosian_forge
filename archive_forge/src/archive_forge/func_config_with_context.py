import asyncio
import threading
from collections import defaultdict
from functools import partial
from itertools import groupby
from typing import (
from langchain_core._api.beta_decorator import beta
from langchain_core.runnables.base import (
from langchain_core.runnables.config import RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output
def config_with_context(config: RunnableConfig, steps: List[Runnable]) -> RunnableConfig:
    """Patch a runnable config with context getters and setters.

    Args:
        config: The runnable config.
        steps: The runnable steps.

    Returns:
        The patched runnable config.
    """
    return _config_with_context(config, steps, _setter, _getter, threading.Event)