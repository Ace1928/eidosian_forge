from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
def load_auth_methods():
    global AUTH_PLUGINS_LOADED
    if AUTH_PLUGINS_LOADED:
        return
    keystone.conf.auth.setup_authentication()
    for plugin in set(CONF.auth.methods):
        AUTH_METHODS[plugin] = load_auth_method(plugin)
    AUTH_PLUGINS_LOADED = True