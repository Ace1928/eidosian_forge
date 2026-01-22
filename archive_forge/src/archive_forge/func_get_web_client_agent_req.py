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
def get_web_client_agent_req(url: str) -> Deferred:
    from twisted.internet import reactor
    from twisted.web.client import Agent
    agent = Agent(reactor)
    return agent.request(b'GET', url.encode('utf-8'))