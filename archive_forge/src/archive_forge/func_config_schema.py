from __future__ import annotations
import asyncio
import collections
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
from contextvars import copy_context
from functools import wraps
from itertools import groupby, tee
from operator import itemgetter
from typing import (
from typing_extensions import Literal, get_args
from langchain_core._api import beta_decorator
from langchain_core.load.dump import dumpd
from langchain_core.load.serializable import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
def config_schema(self, *, include: Optional[Sequence[str]]=None) -> Type[BaseModel]:
    """The type of config this runnable accepts specified as a pydantic model.

        To mark a field as configurable, see the `configurable_fields`
        and `configurable_alternatives` methods.

        Args:
            include: A list of fields to include in the config schema.

        Returns:
            A pydantic model that can be used to validate config.
        """
    include = include or []
    config_specs = self.config_specs
    configurable = create_model('Configurable', **{spec.id: (spec.annotation, Field(spec.default, title=spec.name, description=spec.description)) for spec in config_specs}) if config_specs else None
    return create_model(self.get_name('Config'), **{'configurable': (configurable, None)} if configurable else {}, **{field_name: (field_type, None) for field_name, field_type in RunnableConfig.__annotations__.items() if field_name in [i for i in include if i != 'configurable']})