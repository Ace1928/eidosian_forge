from __future__ import annotations
import asyncio
import inspect
from asyncio import InvalidStateError, Task
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Optional, Union
def ensure_db_setup(self) -> None:
    if self.async_setup_db_task:
        try:
            self.async_setup_db_task.result()
        except InvalidStateError:
            raise ValueError("Asynchronous setup of the DB not finished. NB: AstraDB components sync methods shouldn't be called from the event loop. Consider using their async equivalents.")