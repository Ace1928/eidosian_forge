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
def request_context(func):

    @functools.wraps(func)
    def wrapped(self, ctx, *args, **kwargs):
        try:
            return func(self, ctx, *args, **kwargs)
        except exception.HeatException:
            raise oslo_messaging.rpc.dispatcher.ExpectedException()
    return wrapped