from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str
def download_error(self, failure: Failure, request: Request, spider: Spider, errmsg: Optional[str]=None) -> dict:
    """Logs a download error message from a spider (typically coming from
        the engine).

        .. versionadded:: 2.0
        """
    args: Dict[str, Any] = {'request': request}
    if errmsg:
        msg = DOWNLOADERRORMSG_LONG
        args['errmsg'] = errmsg
    else:
        msg = DOWNLOADERRORMSG_SHORT
    return {'level': logging.ERROR, 'msg': msg, 'args': args}