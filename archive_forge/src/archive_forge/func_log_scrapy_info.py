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
def log_scrapy_info(settings: Settings) -> None:
    logger.info('Scrapy %(version)s started (bot: %(bot)s)', {'version': scrapy.__version__, 'bot': settings['BOT_NAME']})
    versions = [f'{name} {version}' for name, version in scrapy_components_versions() if name != 'Scrapy']
    logger.info('Versions: %(versions)s', {'versions': ', '.join(versions)})