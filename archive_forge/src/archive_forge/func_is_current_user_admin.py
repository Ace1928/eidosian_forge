from __future__ import absolute_import
import os
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import user_service_pb
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def is_current_user_admin():
    """Specifies whether the user making a request is an application admin.

  Because administrator status is not persisted in the datastore,
  `is_current_user_admin()` is a separate function rather than a member function
  of the `User` class. The status only exists for the user making the current
  request.

  Returns:
    `True` if the user is an administrator; all other user types return `False`.
  """
    return os.environ.get('USER_IS_ADMIN', '0') == '1'