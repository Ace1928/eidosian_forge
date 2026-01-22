from __future__ import absolute_import
import os
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import user_service_pb
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def create_logout_url(dest_url, _auth_domain=None):
    """Computes the logout URL and specified destination URL for the request.

  This function works for Google Accounts applications.

  Args:
    dest_url: String that is the desired final destination URL for the user
        after the user has logged out. If `dest_url` does not specify a host,
        the host from the current request is used.

  Returns:
    Logout URL as a string.
  """
    req = user_service_pb.CreateLogoutURLRequest()
    resp = user_service_pb.CreateLogoutURLResponse()
    req.set_destination_url(dest_url)
    if _auth_domain:
        req.set_auth_domain(_auth_domain)
    try:
        apiproxy_stub_map.MakeSyncCall('user', 'CreateLogoutURL', req, resp)
    except apiproxy_errors.ApplicationError as e:
        if e.application_error == user_service_pb.UserServiceError.REDIRECT_URL_TOO_LONG:
            raise RedirectTooLongError
        else:
            raise e
    return resp.logout_url()