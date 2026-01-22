import sys
import logging
from sentry_sdk import utils
from sentry_sdk.hub import Hub
from sentry_sdk.utils import logger
from sentry_sdk.client import _client_init_debug
from logging import LogRecord
def init_debug_support():
    if not logger.handlers:
        configure_logger()
    configure_debug_hub()