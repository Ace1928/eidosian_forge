from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str
def crawled(self, request: Request, response: Response, spider: Spider) -> dict:
    """Logs a message when the crawler finds a webpage."""
    request_flags = f' {str(request.flags)}' if request.flags else ''
    response_flags = f' {str(response.flags)}' if response.flags else ''
    return {'level': logging.DEBUG, 'msg': CRAWLEDMSG, 'args': {'status': response.status, 'request': request, 'request_flags': request_flags, 'referer': referer_str(request), 'response_flags': response_flags, 'flags': response_flags}}