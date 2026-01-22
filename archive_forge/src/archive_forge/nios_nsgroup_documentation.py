from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_NSGROUP
from ..module_utils.api import normalize_ib_spec
entrypoint for module execution.