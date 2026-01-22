from __future__ import absolute_import, division, print_function
import json
import re
import shlex
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command
def find_image_id(self, image_id=None):
    if image_id is None:
        image_id = re.sub(':.*$', '', self.image_name)
    args = ['image', 'ls', '--quiet', '--no-trunc']
    rc, candidates, err = self._run(args, ignore_errors=True)
    candidates = [re.sub('^sha256:', '', c) for c in str.splitlines(candidates)]
    for c in candidates:
        if c.startswith(image_id):
            return image_id
    return None