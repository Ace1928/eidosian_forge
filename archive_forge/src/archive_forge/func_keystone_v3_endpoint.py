import functools
from keystoneauth1 import access
from keystoneauth1.identity import access as access_plugin
from keystoneauth1.identity import generic
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_config import cfg
from oslo_context import context
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import importutils
from heat.common import config
from heat.common import endpoint_utils
from heat.common import exception
from heat.common import policy
from heat.common import wsgi
from heat.engine import clients
@property
def keystone_v3_endpoint(self):
    if self.auth_url:
        return self.auth_url.replace('v2.0', 'v3')
    else:
        auth_uri = endpoint_utils.get_auth_uri()
        if auth_uri:
            return auth_uri
        else:
            LOG.error('Keystone API endpoint not provided. Set auth_uri in section [clients_keystone] of the configuration file.')
            raise exception.AuthorizationFailure()