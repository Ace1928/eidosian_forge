import logging
import re
from logging.config import dictConfig
import threading
from typing import Union
def clear_logger(logger: Union[str, logging.Logger]):
    """Reset a logger, clearing its handlers and enabling propagation.

    Args:
        logger: Logger to be cleared
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    logger.propagate = True
    logger.handlers.clear()