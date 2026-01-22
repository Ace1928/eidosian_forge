from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api import auth
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
def update_plugin(self):
    if self.existing_plugin:
        differences = self.has_different_config()
        if not differences.empty:
            if not self.check_mode:
                try:
                    data = prepare_options(self.parameters.plugin_options)
                    self.client.post_json('/plugins/{0}/set', self.preferred_name, data=data)
                except APIError as e:
                    self.client.fail(to_native(e))
            self.actions.append('Updated plugin %s settings' % self.preferred_name)
            self.changed = True
    else:
        self.client.fail('Cannot update the plugin: Plugin does not exist')