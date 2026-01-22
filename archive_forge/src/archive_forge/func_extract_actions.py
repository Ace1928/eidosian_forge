from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def extract_actions(events):
    actions = []
    pull_actions = set()
    for event in events:
        if event.resource_type == ResourceType.IMAGE_LAYER and event.status in DOCKER_PULL_PROGRESS_WORKING:
            pull_id = (event.resource_id, event.status)
            if pull_id not in pull_actions:
                pull_actions.add(pull_id)
                actions.append({'what': event.resource_type, 'id': event.resource_id, 'status': event.status})
        if event.resource_type != ResourceType.IMAGE_LAYER and event.status in DOCKER_STATUS_WORKING:
            actions.append({'what': event.resource_type, 'id': event.resource_id, 'status': event.status})
    return actions