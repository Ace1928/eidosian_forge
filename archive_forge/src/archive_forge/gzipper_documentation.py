import gzip
import io
from paste.response import header_value, remove_header
from paste.httpheaders import CONTENT_LENGTH

    Wrap the middleware, so that it applies gzipping to a response
    when it is supported by the browser and the content is of
    type ``text/*`` or ``application/*``
    