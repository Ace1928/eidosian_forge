from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def gio_mime_runner(module, **kwargs):
    return CmdRunner(module, command=['gio', 'mime'], arg_formats=dict(mime_type=cmd_runner_fmt.as_list(), handler=cmd_runner_fmt.as_list()), **kwargs)