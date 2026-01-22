from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def get_image_digest(self, name, resolve=False):
    if not name or not resolve:
        return name
    repo, tag = parse_repository_tag(name)
    if not tag:
        tag = 'latest'
    name = repo + ':' + tag
    distribution_data = self.client.inspect_distribution(name)
    digest = distribution_data['Descriptor']['digest']
    return '%s@%s' % (name, digest)