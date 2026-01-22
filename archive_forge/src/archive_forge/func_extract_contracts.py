import re
import sys
from functools import wraps
from inspect import getmembers
from types import CoroutineType
from typing import AsyncGenerator, Dict
from unittest import TestCase
from scrapy.http import Request
from scrapy.utils.python import get_spec
from scrapy.utils.spider import iterate_spider_output
def extract_contracts(self, method):
    contracts = []
    for line in method.__doc__.split('\n'):
        line = line.strip()
        if line.startswith('@'):
            name, args = re.match('@(\\w+)\\s*(.*)', line).groups()
            args = re.split('\\s+', args)
            contracts.append(self.contracts[name](method, *args))
    return contracts