from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def state_disabled(self):
    self._generic_state_action(self.is_snap_enabled, 'snaps_disabled', ['classic', 'channel', 'state'])