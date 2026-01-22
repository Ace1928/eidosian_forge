from __future__ import absolute_import
import os
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import user_service_pb
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def create_login_url(dest_url=None, _auth_domain=None, federated_identity=None):
    """Computes the login URL for redirection.

  Args:
    dest_url: String that is the desired final destination URL for the user
        once login is complete. If `dest_url` does not specify a host, the host
        from the current request is used.
    federated_identity: Decommissioned, don't use. Setting this to a non-None
        value raises a NotAllowedError

  Returns:
       Login URL as a string. The login URL will use Google Accounts.

  Raises:
      NotAllowedError: If federated_identity is not None.
  """
    req = user_service_pb.CreateLoginURLRequest()
    resp = user_service_pb.CreateLoginURLResponse()
    if dest_url:
        req.set_destination_url(dest_url)
    else:
        req.set_destination_url('')
    if _auth_domain:
        req.set_auth_domain(_auth_domain)
    if federated_identity:
        raise NotAllowedError('OpenID 2.0 support is decomissioned')
    try:
        apiproxy_stub_map.MakeSyncCall('user', 'CreateLoginURL', req, resp)
    except apiproxy_errors.ApplicationError as e:
        if e.application_error == user_service_pb.UserServiceError.REDIRECT_URL_TOO_LONG:
            raise RedirectTooLongError
        elif e.application_error == user_service_pb.UserServiceError.NOT_ALLOWED:
            raise NotAllowedError
        else:
            raise e
    return resp.login_url()