from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import sys
import argparse
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.utils.vars import combine_vars
from ansible.utils.display import Display
from ansible.vars.plugins import get_vars_from_inventory_sources, get_vars_from_path
def inventory_graph(self):
    start_at = self._get_group(context.CLIARGS['pattern'])
    if start_at:
        return '\n'.join(self._graph_group(start_at))
    else:
        raise AnsibleOptionsError('Pattern must be valid group name when using --graph')