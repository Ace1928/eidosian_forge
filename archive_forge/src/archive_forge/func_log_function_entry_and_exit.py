import asyncio
import aiofiles
import logging
from typing import Optional, Literal, Union, Any
def log_function_entry_and_exit(logger: logging.Logger):
    """
    A decorator to log the entry and exit of asynchronous functions.

    This decorator logs when a function is entered and exited, providing visibility into the function's execution flow.
    It is particularly useful for debugging asynchronous operations and understanding the sequence of function calls.

    Args:
        logger (logging.Logger): The logger instance used to log the function's entry and exit points.

    Returns:
        The decorated function with added logging capabilities.
    """

    def decorator(func):

        async def wrapper(*args, **kwargs):
            logger.debug(f'Entering: {func.__name__} with args: {args} and kwargs: {kwargs}')
            result = await func(*args, **kwargs)
            logger.debug(f'Exiting: {func.__name__} with result: {result}')
            return result
        return wrapper
    return decorator