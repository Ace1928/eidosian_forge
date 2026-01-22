from __future__ import annotations
import json
import typing as t
from http import HTTPStatus
from urllib.parse import urljoin
from ..datastructures import Headers
from ..http import remove_entity_headers
from ..sansio.response import Response as _SansIOResponse
from ..urls import _invalid_iri_to_uri
from ..urls import iri_to_uri
from ..utils import cached_property
from ..wsgi import ClosingIterator
from ..wsgi import get_current_url
from werkzeug._internal import _get_environ
from werkzeug.http import generate_etag
from werkzeug.http import http_date
from werkzeug.http import is_resource_modified
from werkzeug.http import parse_etags
from werkzeug.http import parse_range_header
from werkzeug.wsgi import _RangeWrapper
def get_wsgi_headers(self, environ: WSGIEnvironment) -> Headers:
    """This is automatically called right before the response is started
        and returns headers modified for the given environment.  It returns a
        copy of the headers from the response with some modifications applied
        if necessary.

        For example the location header (if present) is joined with the root
        URL of the environment.  Also the content length is automatically set
        to zero here for certain status codes.

        .. versionchanged:: 0.6
           Previously that function was called `fix_headers` and modified
           the response object in place.  Also since 0.6, IRIs in location
           and content-location headers are handled properly.

           Also starting with 0.6, Werkzeug will attempt to set the content
           length if it is able to figure it out on its own.  This is the
           case if all the strings in the response iterable are already
           encoded and the iterable is buffered.

        :param environ: the WSGI environment of the request.
        :return: returns a new :class:`~werkzeug.datastructures.Headers`
                 object.
        """
    headers = Headers(self.headers)
    location: str | None = None
    content_location: str | None = None
    content_length: str | int | None = None
    status = self.status_code
    for key, value in headers:
        ikey = key.lower()
        if ikey == 'location':
            location = value
        elif ikey == 'content-location':
            content_location = value
        elif ikey == 'content-length':
            content_length = value
    if location is not None:
        location = _invalid_iri_to_uri(location)
        if self.autocorrect_location_header:
            current_url = get_current_url(environ, strip_querystring=True)
            current_url = iri_to_uri(current_url)
            location = urljoin(current_url, location)
        headers['Location'] = location
    if content_location is not None:
        headers['Content-Location'] = iri_to_uri(content_location)
    if 100 <= status < 200 or status == 204:
        headers.remove('Content-Length')
    elif status == 304:
        remove_entity_headers(headers)
    if self.automatically_set_content_length and self.is_sequence and (content_length is None) and (status not in (204, 304)) and (not 100 <= status < 200):
        content_length = sum((len(x) for x in self.iter_encoded()))
        headers['Content-Length'] = str(content_length)
    return headers