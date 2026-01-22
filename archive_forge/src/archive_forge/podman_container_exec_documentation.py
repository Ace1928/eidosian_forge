from __future__ import absolute_import, division, print_function
import shlex
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command

    Execute podman-container-exec for the given options
    