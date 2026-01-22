from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.core import exceptions as core_exceptions
from six.moves import urllib
def catch_error_raise_cloud_api_error(translation_list, format_str=None, status_code_getter=None):
    """Decorator catches an error and raises CloudApiError with a custom message.

  Args:
    translation_list (list): List of (Exception, int|None, Exception) tuples.
      Translates errors that are instances of first error type to second if the
      status code of the first exception matches the integer value. If the
      status code argument is None, the entry will translate errors of any
      status code.
    format_str (str|None): An api_lib.util.exceptions.FormattableErrorPayload
      format string. Note that any properties that are accessed here are on the
      FormattableErrorPayload object, not the object returned from the server.
    status_code_getter (Exception -> int|None): Function that gets a status code
      from the exception type raised by the underlying client, e.g.
      apitools_exceptions.HttpError. If None, only entries with a null status
      code in `translation_list` will translate errors.

  Returns:
    A decorator that catches errors and raises a CloudApiError with a
      customizable error message.

  Example:
    @catch_error_raise_cloud_api_error(
        [(apitools_exceptions.HttpError, GcsApiError)],
        'Error [{status_code}]')
    def some_func_that_might_throw_an_error():
  """

    def translate_api_error_decorator(function):

        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                core_exceptions.reraise(translate_error(e, translation_list, format_str=format_str, status_code_getter=status_code_getter))
        return wrapper
    return translate_api_error_decorator