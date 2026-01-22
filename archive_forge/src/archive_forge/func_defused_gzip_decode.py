from __future__ import print_function, absolute_import
import io
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden, PY3
def defused_gzip_decode(data, limit=None):
    """gzip encoded data -> unencoded data

    Decode data using the gzip content encoding as described in RFC 1952
    """
    if not gzip:
        raise NotImplementedError
    if limit is None:
        limit = MAX_DATA
    f = io.BytesIO(data)
    gzf = gzip.GzipFile(mode='rb', fileobj=f)
    try:
        if limit < 0:
            decoded = gzf.read()
        else:
            decoded = gzf.read(limit + 1)
    except IOError:
        raise ValueError('invalid data')
    f.close()
    gzf.close()
    if limit >= 0 and len(decoded) > limit:
        raise ValueError('max gzipped payload length exceeded')
    return decoded