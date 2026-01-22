from __future__ import (absolute_import, division, print_function)
import sys
import time
import socket
import array
import json
import time
import re
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict
def vlagConfig(module, prompt, answer):
    retVal = ''
    command = 'vlag '
    vlagArg1 = module.params['vlagArg1']
    vlagArg2 = module.params['vlagArg2']
    vlagArg3 = module.params['vlagArg3']
    vlagArg4 = module.params['vlagArg4']
    deviceType = module.params['deviceType']
    if vlagArg1 == 'enable':
        command = command + vlagArg1 + ' '
    elif vlagArg1 == 'auto-recovery':
        command = command + vlagArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_auto_recovery', vlagArg2)
        if value == 'ok':
            command = command + vlagArg2
        else:
            retVal = 'Error-160'
            return retVal
    elif vlagArg1 == 'config-consistency':
        command = command + vlagArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_config_consistency', vlagArg2)
        if value == 'ok':
            command = command + vlagArg2
        else:
            retVal = 'Error-161'
            return retVal
    elif vlagArg1 == 'isl':
        command = command + vlagArg1 + ' port-channel '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_port_aggregation', vlagArg2)
        if value == 'ok':
            command = command + vlagArg2
        else:
            retVal = 'Error-162'
            return retVal
    elif vlagArg1 == 'mac-address-table':
        command = command + vlagArg1 + ' refresh'
    elif vlagArg1 == 'peer-gateway':
        command = command + vlagArg1 + ' '
    elif vlagArg1 == 'priority':
        command = command + vlagArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_priority', vlagArg2)
        if value == 'ok':
            command = command + vlagArg2
        else:
            retVal = 'Error-163'
            return retVal
    elif vlagArg1 == 'startup-delay':
        command = command + vlagArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_startup_delay', vlagArg2)
        if value == 'ok':
            command = command + vlagArg2
        else:
            retVal = 'Error-164'
            return retVal
    elif vlagArg1 == 'tier-id':
        command = command + vlagArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_tier_id', vlagArg2)
        if value == 'ok':
            command = command + vlagArg2
        else:
            retVal = 'Error-165'
            return retVal
    elif vlagArg1 == 'vrrp':
        command = command + vlagArg1 + ' active'
    elif vlagArg1 == 'instance':
        command = command + vlagArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_instance', vlagArg2)
        if value == 'ok':
            command = command + vlagArg2
            if vlagArg3 is not None:
                command = command + ' port-channel '
                value = cnos.checkSanityofVariable(deviceType, 'vlag_port_aggregation', vlagArg3)
                if value == 'ok':
                    command = command + vlagArg3
                else:
                    retVal = 'Error-162'
                    return retVal
            else:
                command = command + ' enable '
        else:
            retVal = 'Error-166'
            return retVal
    elif vlagArg1 == 'hlthchk':
        command = command + vlagArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'vlag_hlthchk_options', vlagArg2)
        if value == 'ok':
            if vlagArg2 == 'keepalive-attempts':
                value = cnos.checkSanityofVariable(deviceType, 'vlag_keepalive_attempts', vlagArg3)
                if value == 'ok':
                    command = command + vlagArg2 + ' ' + vlagArg3
                else:
                    retVal = 'Error-167'
                    return retVal
            elif vlagArg2 == 'keepalive-interval':
                value = cnos.checkSanityofVariable(deviceType, 'vlag_keepalive_interval', vlagArg3)
                if value == 'ok':
                    command = command + vlagArg2 + ' ' + vlagArg3
                else:
                    retVal = 'Error-168'
                    return retVal
            elif vlagArg2 == 'retry-interval':
                value = cnos.checkSanityofVariable(deviceType, 'vlag_retry_interval', vlagArg3)
                if value == 'ok':
                    command = command + vlagArg2 + ' ' + vlagArg3
                else:
                    retVal = 'Error-169'
                    return retVal
            elif vlagArg2 == 'peer-ip':
                value = cnos.checkSanityofVariable(deviceType, 'vlag_peerip', vlagArg3)
                if value == 'ok':
                    command = command + vlagArg2 + ' ' + vlagArg3
                    if vlagArg4 is not None:
                        value = cnos.checkSanityofVariable(deviceType, 'vlag_peerip_vrf', vlagArg4)
                        if value == 'ok':
                            command = command + ' vrf ' + vlagArg4
                        else:
                            retVal = 'Error-170'
                            return retVal
        else:
            retVal = 'Error-171'
            return retVal
    else:
        retVal = 'Error-172'
        return retVal
    cmd = [{'command': command, 'prompt': None, 'answer': None}]
    retVal = retVal + str(cnos.run_cnos_commands(module, cmd))
    return retVal