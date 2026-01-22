from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def common_compose_argspec():
    return dict(project_src=dict(type='path', required=True), project_name=dict(type='str'), files=dict(type='list', elements='path'), env_files=dict(type='list', elements='path'), profiles=dict(type='list', elements='str'))