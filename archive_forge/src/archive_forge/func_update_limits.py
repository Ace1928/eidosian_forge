from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def update_limits(self, container, container_image, image, host_info):
    limits_differ, different_limits = self.has_different_resource_limits(container, container_image, image, host_info)
    if limits_differ:
        self.log('limit differences:')
        self.log(different_limits.get_legacy_docker_container_diffs(), pretty_print=True)
        self.diff_tracker.merge(different_limits)
    if limits_differ and (not self.check_mode):
        self.container_update(container.id, self._compose_update_parameters())
        return self._get_container(container.id)
    return container