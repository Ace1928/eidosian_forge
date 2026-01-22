from __future__ import absolute_import
from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect
from ..exceptions import HeaderParsingError
from ..packages.six.moves import http_client as httplib
def assert_header_parsing(headers):
    """
    Asserts whether all headers have been successfully parsed.
    Extracts encountered errors from the result of parsing headers.

    Only works on Python 3.

    :param http.client.HTTPMessage headers: Headers to verify.

    :raises urllib3.exceptions.HeaderParsingError:
        If parsing errors are found.
    """
    if not isinstance(headers, httplib.HTTPMessage):
        raise TypeError('expected httplib.Message, got {0}.'.format(type(headers)))
    defects = getattr(headers, 'defects', None)
    get_payload = getattr(headers, 'get_payload', None)
    unparsed_data = None
    if get_payload:
        if not headers.is_multipart():
            payload = get_payload()
            if isinstance(payload, (bytes, str)):
                unparsed_data = payload
    if defects:
        defects = [defect for defect in defects if not isinstance(defect, (StartBoundaryNotFoundDefect, MultipartInvariantViolationDefect))]
    if defects or unparsed_data:
        raise HeaderParsingError(defects=defects, unparsed_data=unparsed_data)