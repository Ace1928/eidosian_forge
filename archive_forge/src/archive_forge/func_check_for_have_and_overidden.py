from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def check_for_have_and_overidden(self, have):
    if have and self.state == 'overridden':
        for name, entry in iteritems(have):
            if name != 'object_type':
                self.addcmd(entry, 'og_name', True)