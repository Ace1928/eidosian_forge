from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def is_change_applicable_for_power_state(current_power_state, apply_power_state):
    """ checks if changes are applicable or not for current system state
        :param current_power_state: Current power state
        :type current_power_state: str
        :param apply_power_state: Required power state
        :type apply_power_state: str
        :return: boolean True if changes is applicable
    """
    on_states = ['On', 'PoweringOn']
    off_states = ['Off', 'PoweringOff']
    reset_map_apply = {('On', 'ForceOn'): off_states, ('PushPowerButton',): on_states + off_states, ('ForceOff', 'ForceRestart', 'GracefulRestart', 'GracefulShutdown', 'Nmi', 'PowerCycle'): on_states}
    is_reset_applicable = False
    for apply_states, applicable_states in reset_map_apply.items():
        if apply_power_state in apply_states:
            if current_power_state in applicable_states:
                is_reset_applicable = True
                break
            break
    return is_reset_applicable