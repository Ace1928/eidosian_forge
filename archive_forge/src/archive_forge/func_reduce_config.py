from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def reduce_config(name, removed_bricks, replicas, force):
    out = run_gluster(['volume', 'heal', name, 'info'])
    summary = out.split('\n')
    for line in summary:
        if 'Number' in line and int(line.split(':')[1].strip()) != 0:
            module.fail_json(msg='Operation aborted, self-heal in progress.')
    args = ['volume', 'remove-brick', name, 'replica', replicas]
    args.extend(removed_bricks)
    if force:
        args.append('force')
    else:
        module.fail_json(msg='Force option is mandatory')
    run_gluster(args)