from scrapy.core.downloader.handlers.http import HTTPDownloadHandler
from scrapy.exceptions import NotConfigured
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import create_instance
def download_request(self, request, spider):
    p = urlparse_cached(request)
    scheme = 'https' if request.meta.get('is_secure') else 'http'
    bucket = p.hostname
    path = p.path + '?' + p.query if p.query else p.path
    url = f'{scheme}://{bucket}.s3.amazonaws.com{path}'
    if self.anon:
        request = request.replace(url=url)
    else:
        import botocore.awsrequest
        awsrequest = botocore.awsrequest.AWSRequest(method=request.method, url=f'{scheme}://s3.amazonaws.com/{bucket}{path}', headers=request.headers.to_unicode_dict(), data=request.body)
        self._signer.add_auth(awsrequest)
        request = request.replace(url=url, headers=awsrequest.headers.items())
    return self._download_http(request, spider)