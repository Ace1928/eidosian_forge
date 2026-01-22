from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def context_config(self, cmd):
    if 'context' not in self._rendered_configuration:
        self._rendered_configuration['context'] = list()
    self._rendered_configuration['context'].extend(to_list(cmd))