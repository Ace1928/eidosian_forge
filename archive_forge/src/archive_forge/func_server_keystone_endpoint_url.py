from keystoneauth1 import session
from heat.common import context
def server_keystone_endpoint_url(self, fallback_endpoint):
    return fallback_endpoint