from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def has_wanted_interfaces(self, wanted, actual):
    """ Compares the interfaces as specified by the user, with the
        interfaces as reported by the server.

        """
    if len(wanted or ()) != len(actual or ()):
        return False

    def match_interface(spec):
        for interface in actual:
            if spec.get('network') == 'public':
                if interface['type'] == 'public':
                    break
            if spec.get('network') is not None:
                if interface['type'] == 'private':
                    if interface['network']['uuid'] == spec['network']:
                        break
            wanted_subnet_ids = set((a['subnet'] for a in spec.get('addresses') or ()))
            actual_subnet_ids = set((a['subnet']['uuid'] for a in interface['addresses']))
            if wanted_subnet_ids == actual_subnet_ids:
                break
        else:
            return False
        for wanted_addr in spec.get('addresses') or ():
            if 'address' not in wanted_addr:
                continue
            addresses = set((a['address'] for a in interface['addresses']))
            if wanted_addr['address'] not in addresses:
                return False
        if spec.get('addresses') == [] and interface['addresses'] != []:
            return False
        if interface['addresses'] == [] and spec.get('addresses') != []:
            return False
        return interface
    for spec in wanted:
        if not match_interface(spec):
            return False
    return True