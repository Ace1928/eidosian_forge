from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_templates import (
def sort_commands(self, index):
    old_cmd = self.commands[index:]
    self.commands = self.commands[0:index]
    self.commands.extend([each for each in old_cmd if 'no' in each] + [each for each in old_cmd if 'no' not in each])