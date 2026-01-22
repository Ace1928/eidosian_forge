from __future__ import absolute_import, division, print_function
from copy import copy
import json
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.ibm.qradar.plugins.module_utils.qradar import (
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ibm.qradar.plugins.modules.qradar_log_sources_management import (
action module