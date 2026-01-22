from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.logging_global.logging_global import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.logging_global import (
def get_logging_data(self, connection):
    return connection.get('show configuration commands | grep syslog')