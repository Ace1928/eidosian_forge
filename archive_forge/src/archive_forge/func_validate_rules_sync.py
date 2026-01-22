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
def validate_rules_sync(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> None:
    """
        Asynchronously validates the inputs to the decorated function based on the provided asynchronous validation rules.
        This method ensures that each argument passed to the function adheres to the predefined rules, enhancing the robustness
        and reliability of the function execution.

        Args:
            func (Callable[..., Awaitable[Any]]): The function being decorated.
            *args (Any): Positional arguments passed to the function.
            **kwargs (Any): Keyword arguments passed to the function.

        Raises:
            AsyncValidationException: If any argument fails to satisfy its corresponding asynchronous validation rule.
        """
    logging.debug(f'Validating async rules for function {func.__name__} with args {args} and kwargs {kwargs}')
    if not hasattr(self, '_bound_arguments_checked'):
        bound_arguments = signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        self._bound_arguments_checked = True
        for arg, value in bound_arguments.arguments.items():
            if arg in self.validation_rules:
                validation_rule = self.validation_rules[arg]
                is_valid = asyncio.run(validation_rule(value)) if asyncio.iscoroutinefunction(validation_rule) else validation_rule(value)
                if not is_valid:
                    raise AsyncValidationException(arg, value, f"Validation failed for argument '{arg}' with value '{value}'")