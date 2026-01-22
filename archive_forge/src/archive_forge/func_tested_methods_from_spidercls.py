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
def tested_methods_from_spidercls(self, spidercls):
    is_method = re.compile('^\\s*@', re.MULTILINE).search
    methods = []
    for key, value in getmembers(spidercls):
        if callable(value) and value.__doc__ and is_method(value.__doc__):
            methods.append(key)
    return methods