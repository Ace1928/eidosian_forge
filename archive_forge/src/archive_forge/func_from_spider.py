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
def from_spider(self, spider, results):
    requests = []
    for method in self.tested_methods_from_spidercls(type(spider)):
        bound_method = spider.__getattribute__(method)
        try:
            requests.append(self.from_method(bound_method, results))
        except Exception:
            case = _create_testcase(bound_method, 'contract')
            results.addError(case, sys.exc_info())
    return requests