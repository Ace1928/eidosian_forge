from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
@classmethod
def from_docker_compose_event(cls, resource_type):
    return {'Network': cls.NETWORK, 'Image': cls.IMAGE, 'Volume': cls.VOLUME, 'Container': cls.CONTAINER}[resource_type]