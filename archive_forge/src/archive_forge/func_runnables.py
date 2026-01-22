import asyncio
from typing import (
from langchain_core.load.dump import dumpd
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import py_anext
@property
def runnables(self) -> Iterator[Runnable[Input, Output]]:
    yield self.runnable
    yield from self.fallbacks