from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def container_remove(self, container_id, link=False, force=False):
    volume_state = not self.param_keep_volumes
    self.log('remove container container:%s v:%s link:%s force%s' % (container_id, volume_state, link, force))
    self.results['actions'].append(dict(removed=container_id, volume_state=volume_state, link=link, force=force))
    self.results['changed'] = True
    if not self.check_mode:
        try:
            self.engine_driver.remove_container(self.client, container_id, remove_volumes=volume_state, link=link, force=force)
        except Exception as exc:
            self.client.fail('Error removing container %s: %s' % (container_id, to_native(exc)))