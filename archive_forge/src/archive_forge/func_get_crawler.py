import asyncio
import os
from importlib import import_module
from pathlib import Path
from posixpath import split
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type
from unittest import TestCase, mock
from twisted.internet.defer import Deferred
from twisted.trial.unittest import SkipTest
from scrapy import Spider
from scrapy.crawler import Crawler
from scrapy.utils.boto import is_botocore_available
def get_crawler(spidercls: Optional[Type[Spider]]=None, settings_dict: Optional[Dict[str, Any]]=None, prevent_warnings: bool=True) -> Crawler:
    """Return an unconfigured Crawler object. If settings_dict is given, it
    will be used to populate the crawler settings with a project level
    priority.
    """
    from scrapy.crawler import CrawlerRunner
    settings: Dict[str, Any] = {}
    if prevent_warnings:
        settings['REQUEST_FINGERPRINTER_IMPLEMENTATION'] = '2.7'
    settings.update(settings_dict or {})
    runner = CrawlerRunner(settings)
    crawler = runner.create_crawler(spidercls or TestSpider)
    crawler._apply_settings()
    return crawler