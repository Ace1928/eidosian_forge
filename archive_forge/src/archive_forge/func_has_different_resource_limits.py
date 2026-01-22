from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def has_different_resource_limits(self, container, container_image, image, host_info):
    differences = DifferenceTracker()
    for options, param_values in self.parameters:
        engine = options.get_engine(self.engine_driver.name)
        if not engine.can_update_value(self.engine_driver.get_api_version(self.client)):
            continue
        self._record_differences(differences, options, param_values, engine, container, container_image, image, host_info)
    has_differences = not differences.empty
    return (has_differences, differences)