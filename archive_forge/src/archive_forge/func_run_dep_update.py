from __future__ import absolute_import, division, print_function
import re
import tempfile
import traceback
import copy
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.helm_args_common import (
def run_dep_update(module, chart_ref):
    """
    Run dependency update
    """
    dep_update = module.get_helm_binary() + ' dependency update ' + chart_ref
    rc, out, err = module.run_helm_command(dep_update)