from collections import namedtuple
import re
import textwrap
import warnings
def create_accept_charset_header(header_value):
    """
    Create an object representing the ``Accept-Charset`` header in a request.

    :param header_value: (``str``) header value
    :return: If `header_value` is ``None``, an :class:`AcceptCharsetNoHeader`
             instance.

             | If `header_value` is a valid ``Accept-Charset`` header, an
               :class:`AcceptCharsetValidHeader` instance.

             | If `header_value` is an invalid ``Accept-Charset`` header, an
               :class:`AcceptCharsetInvalidHeader` instance.
    """
    if header_value is None:
        return AcceptCharsetNoHeader()
    if isinstance(header_value, AcceptCharset):
        return header_value.copy()
    try:
        return AcceptCharsetValidHeader(header_value=header_value)
    except ValueError:
        return AcceptCharsetInvalidHeader(header_value=header_value)