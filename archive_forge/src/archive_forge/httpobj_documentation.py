from typing import Union
from urllib.parse import ParseResult, urlparse
from weakref import WeakKeyDictionary
from scrapy.http import Request, Response
Return urlparse.urlparse caching the result, where the argument can be a
    Request or Response object
    