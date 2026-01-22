import gzip
import io
from paste.response import header_value, remove_header
from paste.httpheaders import CONTENT_LENGTH
def make_gzip_middleware(app, global_conf, compress_level=6):
    """
    Wrap the middleware, so that it applies gzipping to a response
    when it is supported by the browser and the content is of
    type ``text/*`` or ``application/*``
    """
    compress_level = int(compress_level)
    return middleware(app, compress_level=compress_level)