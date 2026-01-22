from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.netvisor.netvisor import run_commands
def booleanArgs(arg, trueString, falseString):
    if arg is True:
        return ' %s ' % trueString
    elif arg is False:
        return ' %s ' % falseString
    else:
        return ''