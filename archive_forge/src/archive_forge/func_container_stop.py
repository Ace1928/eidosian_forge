from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def container_stop(self, container_id):
    if self.param_force_kill:
        self.container_kill(container_id)
        return
    self.results['actions'].append(dict(stopped=container_id, timeout=self.module.params['stop_timeout']))
    self.results['changed'] = True
    if not self.check_mode:
        try:
            self.engine_driver.stop_container(self.client, container_id, self.module.params['stop_timeout'])
        except Exception as exc:
            self.fail('Error stopping container %s: %s' % (container_id, to_native(exc)))