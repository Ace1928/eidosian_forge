from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def get_rendered_configuration(self):
    config = list()
    for section in ('context', 'global'):
        config.extend(self._rendered_configuration.get(section, []))
    return config