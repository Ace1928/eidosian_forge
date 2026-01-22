from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import sys
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import VaultEditor, VaultLib, match_encrypt_secret
from ansible.utils.display import Display
@staticmethod
def format_ciphertext_yaml(b_ciphertext, indent=None, name=None):
    indent = indent or 10
    block_format_var_name = ''
    if name:
        block_format_var_name = '%s: ' % name
    block_format_header = '%s!vault |' % block_format_var_name
    lines = []
    vault_ciphertext = to_text(b_ciphertext)
    lines.append(block_format_header)
    for line in vault_ciphertext.splitlines():
        lines.append('%s%s' % (' ' * indent, line))
    yaml_ciphertext = '\n'.join(lines)
    return yaml_ciphertext