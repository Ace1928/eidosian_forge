from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.core import exceptions as core_exceptions
from six.moves import urllib
def translate_error(error, translation_list, format_str=None, status_code_getter=None):
    """Translates error or returns original error if no matches.

  Note, an error will be translated if it is a child class of a value in
  translation_list. Also, translations earlier in the list take priority.

  Args:
    error (Exception): Error to translate.
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
    Error (Exception). Translated if match. Else, original error.
  """
    if status_code_getter is None:
        status_code_getter = lambda _: None
    for untranslated_error, untranslated_status_code, translated_error in translation_list:
        if isinstance(error, untranslated_error) and (untranslated_status_code is None or status_code_getter is None or status_code_getter(error) == untranslated_status_code):
            return translated_error(error, format_str)
    return error