import copy
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_config import cfg
from oslo_context import context
from glance.api import policy
def get_ksa_client(context):
    """Returns a keystoneauth Adapter using token from context.

    This will return a simple keystoneauth adapter that can be used to
    make requests against a remote service using the token provided
    (and already authenticated) from the user and stored in a
    RequestContext.

    :param context: User request context
    :returns: keystoneauth1 Adapter object
    """
    auth = token_endpoint.Token(CONF.keystone_authtoken.identity_uri, context.auth_token)
    return session.Session(auth=auth)