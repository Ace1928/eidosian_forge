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
def configurable_fields(self, **kwargs: AnyConfigurableField) -> RunnableSerializable[Input, Output]:
    return self.__class__(which=self.which, default=self.default.configurable_fields(**kwargs), alternatives=self.alternatives, default_key=self.default_key, prefix_keys=self.prefix_keys)