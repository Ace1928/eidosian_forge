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
def copy_dst_to_src(diff):
    if diff is None:
        return
    for f, t in [('dst_size', 'src_size'), ('dst_binary', 'src_binary'), ('before_header', 'after_header'), ('before', 'after')]:
        if f in diff:
            diff[t] = diff[f]
        elif t in diff:
            diff.pop(t)