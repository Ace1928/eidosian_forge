from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def names_from_snaps(self, snaps):

    def process_one(rc, out, err):
        res = [line for line in out.split('\n') if line.startswith('name:')]
        name = res[0].split()[1]
        return [name]

    def process_many(rc, out, err):
        outputs = out.split('\n---')
        res = []
        for sout in outputs:
            res.extend(process_one(rc, sout, ''))
        return res

    def process(rc, out, err):
        if len(snaps) == 1:
            check_error = err
            process_ = process_one
        else:
            check_error = out
            process_ = process_many
        if 'warning: no snap found' in check_error:
            self.do_raise('Snaps not found: {0}.'.format([x.split()[-1] for x in out.split('\n') if x.startswith('warning: no snap found')]))
        return process_(rc, out, err)
    names = []
    if snaps:
        with self.runner('info name', output_process=process) as ctx:
            try:
                names = ctx.run(name=snaps)
            finally:
                self.vars.snapinfo_run_info.append(ctx.run_info)
    return names