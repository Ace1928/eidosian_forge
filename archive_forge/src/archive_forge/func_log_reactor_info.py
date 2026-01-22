from __future__ import annotations
import logging
import sys
import warnings
from logging.config import dictConfig
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union, cast
from twisted.python import log as twisted_log
from twisted.python.failure import Failure
import scrapy
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.settings import Settings
from scrapy.utils.versions import scrapy_components_versions
def log_reactor_info() -> None:
    from twisted.internet import reactor
    logger.debug('Using reactor: %s.%s', reactor.__module__, reactor.__class__.__name__)
    from twisted.internet import asyncioreactor
    if isinstance(reactor, asyncioreactor.AsyncioSelectorReactor):
        logger.debug('Using asyncio event loop: %s.%s', reactor._asyncioEventloop.__module__, reactor._asyncioEventloop.__class__.__name__)