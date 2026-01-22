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
def bgpConfig(module, cmd, prompt, answer):
    retVal = ''
    command = ''
    bgpArg1 = module.params['bgpArg1']
    bgpArg2 = module.params['bgpArg2']
    bgpArg3 = module.params['bgpArg3']
    bgpArg4 = module.params['bgpArg4']
    bgpArg5 = module.params['bgpArg5']
    bgpArg6 = module.params['bgpArg6']
    bgpArg7 = module.params['bgpArg7']
    bgpArg8 = module.params['bgpArg8']
    asNum = module.params['asNum']
    deviceType = module.params['deviceType']
    if bgpArg1 == 'address-family':
        command = command + bgpArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_address_family', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2 + ' ' + 'unicast \n'
            inner_cmd = [{'command': command, 'prompt': None, 'answer': None}]
            cmd.extend(inner_cmd)
            retVal = retVal + bgpAFConfig(module, cmd, prompt, answer)
            return retVal
        else:
            retVal = 'Error-178'
            return retVal
    elif bgpArg1 == 'bestpath':
        command = command + bgpArg1 + ' '
        if bgpArg2 == 'always-compare-med':
            command = command + bgpArg2
        elif bgpArg2 == 'compare-confed-aspath':
            command = command + bgpArg2
        elif bgpArg2 == 'compare-routerid':
            command = command + bgpArg2
        elif bgpArg2 == 'dont-compare-originator-id':
            command = command + bgpArg2
        elif bgpArg2 == 'tie-break-on-age':
            command = command + bgpArg2
        elif bgpArg2 == 'as-path':
            command = command + bgpArg2 + ' '
            if bgpArg3 == 'ignore' or bgpArg3 == 'multipath-relax':
                command = command + bgpArg3
            else:
                retVal = 'Error-179'
                return retVal
        elif bgpArg2 == 'med':
            command = command + bgpArg2 + ' '
            if bgpArg3 == 'confed' or bgpArg3 == 'missing-as-worst' or bgpArg3 == 'non-deterministic' or (bgpArg3 == 'remove-recv-med') or (bgpArg3 == 'remove-send-med'):
                command = command + bgpArg3
            else:
                retVal = 'Error-180'
                return retVal
        else:
            retVal = 'Error-181'
            return retVal
    elif bgpArg1 == 'bgp':
        command = command + bgpArg1 + ' as-local-count '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_bgp_local_count', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2
        else:
            retVal = 'Error-182'
            return retVal
    elif bgpArg1 == 'cluster-id':
        command = command + bgpArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'cluster_id_as_ip', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2
        else:
            value = cnos.checkSanityofVariable(deviceType, 'cluster_id_as_number', bgpArg2)
            if value == 'ok':
                command = command + bgpArg2
            else:
                retVal = 'Error-183'
                return retVal
    elif bgpArg1 == 'confederation':
        command = command + bgpArg1 + ' '
        if bgpArg2 == 'identifier':
            value = cnos.checkSanityofVariable(deviceType, 'confederation_identifier', bgpArg3)
            if value == 'ok':
                command = command + bgpArg2 + ' ' + bgpArg3 + '\n'
            else:
                retVal = 'Error-184'
                return retVal
        elif bgpArg2 == 'peers':
            value = cnos.checkSanityofVariable(deviceType, 'confederation_peers_as', bgpArg3)
            if value == 'ok':
                command = command + bgpArg2 + ' ' + bgpArg3
            else:
                retVal = 'Error-185'
                return retVal
        else:
            retVal = 'Error-186'
            return retVal
    elif bgpArg1 == 'enforce-first-as':
        command = command + bgpArg1
    elif bgpArg1 == 'fast-external-failover':
        command = command + bgpArg1
    elif bgpArg1 == 'graceful-restart':
        command = command + bgpArg1 + ' stalepath-time '
        value = cnos.checkSanityofVariable(deviceType, 'stalepath_delay_value', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2
        else:
            retVal = 'Error-187'
            return retVal
    elif bgpArg1 == 'graceful-restart-helper':
        command = command + bgpArg1
    elif bgpArg1 == 'log-neighbor-changes':
        command = command + bgpArg1
    elif bgpArg1 == 'maxas-limit':
        command = command + bgpArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'maxas_limit_as', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2
        else:
            retVal = 'Error-188'
            return retVal
    elif bgpArg1 == 'neighbor':
        command = command + bgpArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'neighbor_ipaddress', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2
            if bgpArg3 is not None:
                command = command + ' remote-as '
                value = cnos.checkSanityofVariable(deviceType, 'neighbor_as', bgpArg3)
                if value == 'ok':
                    command = command + bgpArg3
                    inner_cmd = [{'command': command, 'prompt': None, 'answer': None}]
                    cmd.extend(inner_cmd)
                    retVal = retVal + bgpNeighborConfig(module, cmd, prompt, answer)
                    return retVal
        else:
            retVal = 'Error-189'
            return retVal
    elif bgpArg1 == 'router-id':
        command = command + bgpArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'router_id', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2
        else:
            retVal = 'Error-190'
            return retVal
    elif bgpArg1 == 'shutdown':
        command = command + bgpArg1
    elif bgpArg1 == 'synchronization':
        command = command + bgpArg1
    elif bgpArg1 == 'timers':
        command = command + bgpArg1 + ' bgp '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_keepalive_interval', bgpArg2)
        if value == 'ok':
            command = command + bgpArg2
        else:
            retVal = 'Error-191'
            return retVal
        if bgpArg3 is not None:
            value = cnos.checkSanityofVariable(deviceType, 'bgp_holdtime', bgpArg3)
            if value == 'ok':
                command = command + ' ' + bgpArg3
            else:
                retVal = 'Error-192'
                return retVal
        else:
            retVal = 'Error-192'
            return retVal
    elif bgpArg1 == 'vrf':
        command = command + bgpArg1 + ' default'
    else:
        retVal = 'Error-192'
        return retVal
    inner_cmd = [{'command': command, 'prompt': None, 'answer': None}]
    cmd.extend(inner_cmd)
    retVal = retVal + str(cnos.run_cnos_commands(module, cmd))
    command = 'exit \n'
    return retVal