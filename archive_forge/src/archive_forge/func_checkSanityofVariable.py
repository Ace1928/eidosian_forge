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
def checkSanityofVariable(deviceType, variableId, variableValue):
    retVal = ''
    ruleFile = 'dictionary/' + deviceType + '_rules.lvo'
    ruleString = getRuleStringForVariable(deviceType, ruleFile, variableId)
    retVal = validateValueAgainstRule(ruleString, variableValue)
    return retVal