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
def load_auth_method(method):
    plugin_name = CONF.auth.get(method) or 'default'
    namespace = 'keystone.auth.%s' % method
    driver_manager = _get_auth_driver_manager(namespace, plugin_name)
    return driver_manager.driver