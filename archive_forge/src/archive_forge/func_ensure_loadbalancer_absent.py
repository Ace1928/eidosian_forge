from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_loadbalancer_absent(self, name, alias, location):
    """
        Checks to see if a load balancer exists and deletes it if it does
        :param name: Name of the load balancer
        :param alias: Alias of account
        :param location: Datacenter
        :return: (changed, result)
            changed: Boolean whether a change was made
            result: The result from the CLC API Call
        """
    changed = False
    result = name
    lb_exists = self._loadbalancer_exists(name=name)
    if lb_exists:
        if not self.module.check_mode:
            result = self.delete_loadbalancer(alias=alias, location=location, name=name)
        changed = True
    return (changed, result)