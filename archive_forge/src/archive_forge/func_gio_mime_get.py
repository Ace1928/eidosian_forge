from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def gio_mime_get(runner, mime_type):

    def process(rc, out, err):
        if err.startswith('No default applications for'):
            return None
        out = out.splitlines()[0]
        return out.split()[-1]
    with runner('mime_type', output_process=process) as ctx:
        return ctx.run(mime_type=mime_type)