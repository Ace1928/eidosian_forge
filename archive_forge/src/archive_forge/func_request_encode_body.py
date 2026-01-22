from __future__ import absolute_import
from .filepost import encode_multipart_formdata
from .packages.six.moves.urllib.parse import urlencode
def request_encode_body(self, method, url, fields=None, headers=None, encode_multipart=True, multipart_boundary=None, **urlopen_kw):
    """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the body. This is useful for request methods like POST, PUT, PATCH, etc.

        When ``encode_multipart=True`` (default), then
        :func:`urllib3.encode_multipart_formdata` is used to encode
        the payload with the appropriate content type. Otherwise
        :func:`urllib.parse.urlencode` is used with the
        'application/x-www-form-urlencoded' content type.

        Multipart encoding must be used when posting files, and it's reasonably
        safe to use it in other times too. However, it may break request
        signing, such as with OAuth.

        Supports an optional ``fields`` parameter of key/value strings AND
        key/filetuple. A filetuple is a (filename, data, MIME type) tuple where
        the MIME type is optional. For example::

            fields = {
                'foo': 'bar',
                'fakefile': ('foofile.txt', 'contents of foofile'),
                'realfile': ('barfile.txt', open('realfile').read()),
                'typedfile': ('bazfile.bin', open('bazfile').read(),
                              'image/jpeg'),
                'nonamefile': 'contents of nonamefile field',
            }

        When uploading a file, providing a filename (the first parameter of the
        tuple) is optional but recommended to best mimic behavior of browsers.

        Note that if ``headers`` are supplied, the 'Content-Type' header will
        be overwritten because it depends on the dynamic random boundary string
        which is used to compose the body of the request. The random boundary
        string can be explicitly set with the ``multipart_boundary`` parameter.
        """
    if headers is None:
        headers = self.headers
    extra_kw = {'headers': {}}
    if fields:
        if 'body' in urlopen_kw:
            raise TypeError("request got values for both 'fields' and 'body', can only specify one.")
        if encode_multipart:
            body, content_type = encode_multipart_formdata(fields, boundary=multipart_boundary)
        else:
            body, content_type = (urlencode(fields), 'application/x-www-form-urlencoded')
        extra_kw['body'] = body
        extra_kw['headers'] = {'Content-Type': content_type}
    extra_kw['headers'].update(headers)
    extra_kw.update(urlopen_kw)
    return self.urlopen(method, url, **extra_kw)