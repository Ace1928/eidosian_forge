from scrapy.core.downloader.handlers.http import HTTPDownloadHandler
from scrapy.exceptions import NotConfigured
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import create_instance
@classmethod
def from_crawler(cls, crawler, **kwargs):
    return cls(crawler.settings, crawler=crawler, **kwargs)