from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def container_kill(self, container_id):
    self.results['actions'].append(dict(killed=container_id, signal=self.param_kill_signal))
    self.results['changed'] = True
    if not self.check_mode:
        try:
            self.engine_driver.kill_container(self.client, container_id, kill_signal=self.param_kill_signal)
        except Exception as exc:
            self.fail('Error killing container %s: %s' % (container_id, to_native(exc)))