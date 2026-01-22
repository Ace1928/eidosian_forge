import logging
import os
import pathlib
import sys
import time
import pytest
def configured_string_logging(unique_code, formatter=None):
    """
    Helper function provides logger configured to write to log_output.
    """
    from io import StringIO
    log_output = StringIO()
    handler = logging.StreamHandler(stream=log_output)
    if formatter:
        handler.setFormatter(formatter)
    logger = logging.getLogger('tests.%s' % unique_code)
    logger.setLevel(9)
    logger.propagate = False
    assert not logger.hasHandlers(), 'Must use unique code between tests.'
    logger.addHandler(handler)
    return (logger, log_output)