from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.pipx import pipx_runner
from ansible.module_utils.facts.compat import ansible_facts
def state_upgrade(self):
    if not self.vars.application:
        self.do_raise('Trying to upgrade a non-existent application: {0}'.format(self.vars.name))
    if self.vars.force:
        self.changed = True
    with self.runner('state include_injected index_url force editable pip_args name', check_mode_skip=True) as ctx:
        ctx.run()
        self._capture_results(ctx)