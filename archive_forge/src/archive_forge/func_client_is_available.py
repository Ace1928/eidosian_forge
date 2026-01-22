import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import enabled
from heat.common import exception
from heat.common.i18n import _
from heat.common import pluginutils
def client_is_available(client_plugin):
    if not hasattr(client_plugin.plugin, 'is_available'):
        return True
    return client_plugin.plugin.is_available()