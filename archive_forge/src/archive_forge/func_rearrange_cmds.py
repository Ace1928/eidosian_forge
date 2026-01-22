from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def rearrange_cmds(aces):
    non_negates = []
    negates = []
    for ace in aces:
        if ace.startswith('no'):
            negates.append(ace)
        else:
            non_negates.append(ace)
    if non_negates or negates:
        negates.extend(non_negates)
    return negates