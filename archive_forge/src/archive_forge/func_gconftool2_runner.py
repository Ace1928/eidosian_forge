from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def gconftool2_runner(module, **kwargs):
    return CmdRunner(module, command='gconftool-2', arg_formats=dict(state=cmd_runner_fmt.as_map(_state_map), key=cmd_runner_fmt.as_list(), value_type=cmd_runner_fmt.as_opt_val('--type'), value=cmd_runner_fmt.as_list(), direct=cmd_runner_fmt.as_bool('--direct'), config_source=cmd_runner_fmt.as_opt_val('--config-source')), **kwargs)