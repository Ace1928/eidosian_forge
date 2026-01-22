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
def from_method(self, method, results):
    contracts = self.extract_contracts(method)
    if contracts:
        request_cls = Request
        for contract in contracts:
            if contract.request_cls is not None:
                request_cls = contract.request_cls
        args, kwargs = get_spec(request_cls.__init__)
        kwargs['dont_filter'] = True
        kwargs['callback'] = method
        for contract in contracts:
            kwargs = contract.adjust_request_args(kwargs)
        args.remove('self')
        if set(args).issubset(set(kwargs)):
            request = request_cls(**kwargs)
            for contract in reversed(contracts):
                request = contract.add_pre_hook(request, results)
            for contract in contracts:
                request = contract.add_post_hook(request, results)
            self._clean_req(request, method, results)
            return request