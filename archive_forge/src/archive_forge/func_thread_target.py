from __future__ import annotations
import asyncio
import threading
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.tools.ainetwork.utils import authenticate
def thread_target() -> None:
    nonlocal result_container
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        result_container.append(new_loop.run_until_complete(self._arun(*args, **kwargs)))
    except Exception as e:
        result_container.append(e)
    finally:
        new_loop.close()