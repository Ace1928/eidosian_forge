import collections
import logging
import pytest
import modin.logging
from modin.config import LogMode
def mock_get_logger(ctx):
    ctx.setattr(logging, 'getLogger', _get_logger)