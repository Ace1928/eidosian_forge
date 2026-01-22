from __future__ import absolute_import, division, print_function
import re
import tempfile
import traceback
import copy
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.helm_args_common import (
def fetch_chart_info(module, command, chart_ref):
    """
    Get chart info
    """
    inspect_command = command + ' show chart ' + chart_ref
    rc, out, err = module.run_helm_command(inspect_command)
    return yaml.safe_load(out)