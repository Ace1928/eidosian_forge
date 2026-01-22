import sys
import logging
from sentry_sdk import utils
from sentry_sdk.hub import Hub
from sentry_sdk.utils import logger
from sentry_sdk.client import _client_init_debug
from logging import LogRecord
def configure_logger():
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(' [sentry] %(levelname)s: %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(_HubBasedClientFilter())