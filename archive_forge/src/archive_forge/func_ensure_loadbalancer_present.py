from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_loadbalancer_present(self, name, alias, location, description, status):
    """
        Checks to see if a load balancer exists and creates one if it does not.
        :param name: Name of loadbalancer
        :param alias: Alias of account
        :param location: Datacenter
        :param description: Description of loadbalancer
        :param status: Enabled / Disabled
        :return: (changed, result, lb_id)
            changed: Boolean whether a change was made
            result: The result object from the CLC load balancer request
            lb_id: The load balancer id
        """
    changed = False
    result = name
    lb_id = self._loadbalancer_exists(name=name)
    if not lb_id:
        if not self.module.check_mode:
            result = self.create_loadbalancer(name=name, alias=alias, location=location, description=description, status=status)
            lb_id = result.get('id')
        changed = True
    return (changed, result, lb_id)