import contextlib
import logging
import logging.config
import re
import sys
@contextlib.contextmanager
def suppress_logs(module: str, filter_regex: re.Pattern):
    """
    Context manager that suppresses log messages from the specified module that match the specified
    regular expression. This is useful for suppressing expected log messages from third-party
    libraries that are not relevant to the current test.
    """
    logger = logging.getLogger(module)
    filter = LoggerMessageFilter(module=module, filter_regex=filter_regex)
    logger.addFilter(filter)
    try:
        yield
    finally:
        logger.removeFilter(filter)