from keystoneauth1 import exceptions as keystone_exceptions
from keystoneauth1 import session
from webob import exc
from heat.common import config
from heat.common import context
def auth_filter(app):
    return KeystonePasswordAuthProtocol(app, conf)