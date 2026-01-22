import logging
import logging.config
import logging.handlers
import sys
import time
import asyncio
import aiofiles
from typing import (
import pathlib
import json
from concurrent.futures import Executor, ThreadPoolExecutor
import functools
from functools import wraps
import tracemalloc
import inspect
from inspect import signature, Parameter
from IndegoValidation import AsyncValidationException, ValidationRules
def log_performance_sync(self, func_name: str, start_time: float, end_time: float) -> None:
    """
        Logs the performance of the decorated function, including the time taken for execution.

        Args:
            func_name (str): The name of the function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.
        """
    adjusted_time = end_time - start_time
    logging.debug(f'{func_name} executed in {adjusted_time:.6f}s')