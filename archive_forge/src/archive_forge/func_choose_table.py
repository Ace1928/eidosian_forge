from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.resource.config_backends import base
def choose_table(self, sensitive):
    if sensitive:
        return SensitiveConfig
    else:
        return WhiteListedConfig