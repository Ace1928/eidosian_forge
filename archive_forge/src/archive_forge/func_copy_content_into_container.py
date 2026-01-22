from __future__ import absolute_import, division, print_function
import base64
import io
import os
import stat
import traceback
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.copy import (
from ansible_collections.community.docker.plugins.module_utils._scramble import generate_insecure_key, scramble
def copy_content_into_container(client, container, content, container_path, follow_links, owner_id, group_id, mode, force=False, diff=False, max_file_size_for_diff=1):
    if diff:
        diff = {}
    else:
        diff = None
    container_path, mode, idempotent = is_content_idempotent(client, container, content, container_path, follow_links, owner_id, group_id, mode, force=force, diff=diff, max_file_size_for_diff=max_file_size_for_diff)
    changed = not idempotent
    if changed and (not client.module.check_mode):
        put_file_content(client, container, content=content, out_path=container_path, user_id=owner_id, group_id=group_id, mode=mode)
    result = dict(container_path=container_path, changed=changed)
    if diff:
        key = generate_insecure_key()
        diff['scrambled_diff'] = base64.b64encode(key)
        for k in ('before', 'after'):
            if k in diff:
                diff[k] = scramble(diff[k], key)
        result['diff'] = diff
    client.module.exit_json(**result)