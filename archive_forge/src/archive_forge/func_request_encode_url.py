from __future__ import absolute_import
from .filepost import encode_multipart_formdata
from .packages.six.moves.urllib.parse import urlencode
def request_encode_url(self, method, url, fields=None, headers=None, **urlopen_kw):
    """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the url. This is useful for request methods like GET, HEAD, DELETE, etc.
        """
    if headers is None:
        headers = self.headers
    extra_kw = {'headers': headers}
    extra_kw.update(urlopen_kw)
    if fields:
        url += '?' + urlencode(fields)
    return self.urlopen(method, url, **extra_kw)