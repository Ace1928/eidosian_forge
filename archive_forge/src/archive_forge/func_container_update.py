from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def container_update(self, container_id, update_parameters):
    if update_parameters:
        self.log('update container %s' % container_id)
        self.log(update_parameters, pretty_print=True)
        self.results['actions'].append(dict(updated=container_id, update_parameters=update_parameters))
        self.results['changed'] = True
        if not self.check_mode:
            try:
                self.engine_driver.update_container(self.client, container_id, update_parameters)
            except Exception as exc:
                self.fail('Error updating container %s: %s' % (container_id, to_native(exc)))
    return self._get_container(container_id)