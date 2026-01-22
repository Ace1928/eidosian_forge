from oslo_config import cfg
from saharaclient.api import base as sahara_base
from saharaclient import client as sahara_client
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
def validate_hadoop_version(self, plugin_name, hadoop_version):
    plugin = self.client().plugins.get(plugin_name)
    allowed_versions = plugin.versions
    if hadoop_version not in allowed_versions:
        msg = _("Requested plugin '%(plugin)s' doesn't support version '%(version)s'. Allowed versions are %(allowed)s") % {'plugin': plugin_name, 'version': hadoop_version, 'allowed': ', '.join(allowed_versions)}
        raise exception.StackValidationFailed(message=msg)