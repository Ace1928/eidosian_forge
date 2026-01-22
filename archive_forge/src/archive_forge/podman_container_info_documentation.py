from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
Inspect each container in a cycle in case some of them don't exist.

    Arguments:
        module {AnsibleModule} -- instance of AnsibleModule
        executable {string} -- binary to execute when inspecting containers
        name {list} -- list of containers names to inspect

    Returns:
        list of containers info, stdout as empty, stderr
    