import asyncio
import cProfile
import functools
from functools import wraps
import gc
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from inspect import signature, Parameter
from pathlib import Path
from typing import (
import aiofiles
import msgpack
import zstandard
import pstats
from aiokeydb import AsyncKeyDB, KeyDBClient
from lazyops.utils import logger
from pydantic import BaseModel, ValidationError
import traceback
def with_model(self, model: Type[BaseModel]) -> Callable:
    """
        Decorator to attach a Pydantic model to the caching decorator for result validation.

        Args:
            model (Type[BaseModel]): The Pydantic model to use for result validation.

        Returns:
            Callable: The decorator with the attached Pydantic model.
        """

    def decorator(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            wrapped = self(func)
            wrapped.model = model
            return await wrapped(*args, **kwargs)
        return wrapper
    return decorator