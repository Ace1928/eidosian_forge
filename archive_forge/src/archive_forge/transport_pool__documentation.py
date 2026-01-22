from __future__ import absolute_import
from __future__ import print_function
import threading
import httplib2
from six.moves import range  # pylint: disable=redefined-builtin
This awaits a transport and delegates the request call.

    Args:
      *args: arguments to request.
      **kwargs: named arguments to request.

    Returns:
      tuple of response and content.
    