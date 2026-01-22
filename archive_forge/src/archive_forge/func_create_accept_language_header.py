from collections import namedtuple
import re
import textwrap
import warnings
def create_accept_language_header(header_value):
    """
    Create an object representing the ``Accept-Language`` header in a request.

    :param header_value: (``str``) header value
    :return: If `header_value` is ``None``, an :class:`AcceptLanguageNoHeader`
             instance.

             | If `header_value` is a valid ``Accept-Language`` header, an
               :class:`AcceptLanguageValidHeader` instance.

             | If `header_value` is an invalid ``Accept-Language`` header, an
               :class:`AcceptLanguageInvalidHeader` instance.
    """
    if header_value is None:
        return AcceptLanguageNoHeader()
    if isinstance(header_value, AcceptLanguage):
        return header_value.copy()
    try:
        return AcceptLanguageValidHeader(header_value=header_value)
    except ValueError:
        return AcceptLanguageInvalidHeader(header_value=header_value)