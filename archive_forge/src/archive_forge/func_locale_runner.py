from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def locale_runner(module):
    runner = CmdRunner(module, command=['locale', '-a'], check_rc=True)
    return runner