import asyncio
import atexit
import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener
from typing import Dict, List, Optional, Tuple, Union
def queuify_logger(logger: Union[logging.Logger, str], queue_handler: QueueHandler, queue_listener: QueueListener) -> None:
    """Replace logger's handlers with a queue handler while adding existing
    handlers to a queue listener.

    This is useful when you want to use a default logging config but then
    optionally add a logger's handlers to a queue during runtime.

    Args:
        logger (mixed): Logger instance or string name of logger to queue-ify
            handlers.
        queue_handler (QueueHandler): Instance of a ``QueueHandler``.
        queue_listener (QueueListener): Instance of a ``QueueListener``.

    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    handlers = [handler for handler in logger.handlers if handler not in queue_listener.handlers]
    if handlers:
        queue_listener.handlers = tuple(list(queue_listener.handlers) + handlers)
    del logger.handlers[:]
    logger.addHandler(queue_handler)