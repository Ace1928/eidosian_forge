import logging
import re
from typing import TYPE_CHECKING, Any
from scrapy.http import Request, XmlResponse
from scrapy.spiders import Spider
from scrapy.utils._compression import _DecompressionMaxSizeExceeded
from scrapy.utils.gz import gunzip, gzip_magic_number
from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
def iterloc(it, alt=False):
    for d in it:
        yield d['loc']
        if alt and 'alternate' in d:
            yield from d['alternate']