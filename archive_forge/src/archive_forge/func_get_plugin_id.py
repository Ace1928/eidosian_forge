from oslo_config import cfg
from saharaclient.api import base as sahara_base
from saharaclient import client as sahara_client
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_plugin_id(self, plugin_name):
    """Get the id for the specified plugin name.

        :param plugin_name: the name of the plugin to find
        :returns: the id of :plugin:
        :raises exception.EntityNotFound:
        """
    try:
        self.client().plugins.get(plugin_name)
    except sahara_base.APIException:
        raise exception.EntityNotFound(entity='Plugin', name=plugin_name)