from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def get_diff_state(self, image):
    if not image:
        return dict(exists=False)
    return dict(exists=True, id=image['Id'], tags=sorted(image.get('RepoTags') or []), digests=sorted(image.get('RepoDigests') or []))