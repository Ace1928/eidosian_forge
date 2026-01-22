from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str
def item_error(self, item: Any, exception: BaseException, response: Response, spider: Spider) -> dict:
    """Logs a message when an item causes an error while it is passing
        through the item pipeline.

        .. versionadded:: 2.0
        """
    return {'level': logging.ERROR, 'msg': ITEMERRORMSG, 'args': {'item': item}}