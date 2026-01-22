from __future__ import absolute_import, division, print_function
import time
import socket
import re
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, EntityCollection
from ansible.module_utils.connection import Connection, exec_command
from ansible.module_utils.connection import ConnectionError
def getRuleStringForVariable(deviceType, ruleFile, variableId):
    retVal = ''
    try:
        f = open(ruleFile, 'r')
        for line in f:
            if ':' in line:
                data = line.split(':')
                if data[0].strip() == variableId:
                    retVal = line
    except Exception:
        ruleString = cnos_devicerules.getRuleString(deviceType, variableId)
        retVal = ruleString.strip()
    return retVal