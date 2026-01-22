from typing import Generator, Tuple
from urllib.parse import urljoin
from scrapy.exceptions import NotSupported
from scrapy.http.common import obsolete_setter
from scrapy.http.headers import Headers
from scrapy.http.request import Request
from scrapy.link import Link
from scrapy.utils.trackref import object_ref
@property
def cb_kwargs(self):
    try:
        return self.request.cb_kwargs
    except AttributeError:
        raise AttributeError('Response.cb_kwargs not available, this response is not tied to any request')