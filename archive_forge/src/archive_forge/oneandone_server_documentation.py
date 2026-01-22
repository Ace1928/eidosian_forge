from __future__ import absolute_import, division, print_function
import os
import time
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (

    Allow the incremental count in the description when defined with the
    string formatting (%) operator. Otherwise, repeat the same description.
    