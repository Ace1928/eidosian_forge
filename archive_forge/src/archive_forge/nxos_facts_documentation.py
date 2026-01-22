from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.argspec.facts.facts import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts

    Main entry point for module execution

    :returns: ansible_facts
    