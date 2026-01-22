from __future__ import (absolute_import, division, print_function)
import base64
import datetime
import io
import json
import os
import os.path
import shutil
import stat
import tarfile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, NotFound
def put_file_content(client, container, content, out_path, user_id, group_id, mode, user_name=None):
    """Transfer a file from local to Docker container."""
    out_dir, out_file = os.path.split(out_path)
    stream = _regular_content_tar_generator(content, out_file, user_id, group_id, mode, user_name=user_name)
    ok = _put_archive(client, container, out_dir, stream)
    if not ok:
        raise DockerUnexpectedError('Unknown error while creating file "{0}" in container "{1}".'.format(out_path, container))