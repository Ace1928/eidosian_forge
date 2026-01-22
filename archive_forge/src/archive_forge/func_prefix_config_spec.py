from __future__ import annotations
import enum
import threading
from abc import abstractmethod
from functools import wraps
from typing import (
from weakref import WeakValueDictionary
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
def prefix_config_spec(spec: ConfigurableFieldSpec, prefix: str) -> ConfigurableFieldSpec:
    """Prefix the id of a ConfigurableFieldSpec.

    This is useful when a RunnableConfigurableAlternatives is used as a
    ConfigurableField of another RunnableConfigurableAlternatives.

    Args:
        spec: The ConfigurableFieldSpec to prefix.
        prefix: The prefix to add.

    Returns:

    """
    return ConfigurableFieldSpec(id=f'{prefix}/{spec.id}', name=spec.name, description=spec.description, annotation=spec.annotation, default=spec.default, is_shared=spec.is_shared) if not spec.is_shared else spec