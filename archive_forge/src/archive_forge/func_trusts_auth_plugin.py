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
def trusts_auth_plugin(self):
    if not self._trusts_auth_plugin:
        self._trusts_auth_plugin = ks_loading.load_auth_from_conf_options(cfg.CONF, TRUSTEE_CONF_GROUP, trust_id=self.trust_id)
    if not self._trusts_auth_plugin:
        LOG.error('Please add the trustee credentials you need to the %s section of your heat.conf file.', TRUSTEE_CONF_GROUP)
        raise exception.AuthorizationFailure()
    return self._trusts_auth_plugin