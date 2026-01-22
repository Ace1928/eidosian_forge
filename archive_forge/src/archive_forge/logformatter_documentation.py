from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str
Logs a download error message from a spider (typically coming from
        the engine).

        .. versionadded:: 2.0
        