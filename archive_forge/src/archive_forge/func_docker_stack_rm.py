from __future__ import (absolute_import, division, print_function)
import json
import os
import tempfile
import traceback
from ansible.module_utils.six import string_types
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
def docker_stack_rm(client, stack_name, retries, interval):
    command = ['stack', 'rm', stack_name]
    rc, out, err = client.call_cli(*command)
    while to_native(err) != 'Nothing found in stack: %s\n' % stack_name and retries > 0:
        sleep(interval)
        retries = retries - 1
        rc, out, err = client.call_cli(*command)
    return (rc, to_native(out), to_native(err))