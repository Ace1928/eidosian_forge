from __future__ import (absolute_import, division, print_function)
import locale
import os
import sys
from importlib.metadata import version
from ansible.module_utils.compat.version import LooseVersion
import errno
import getpass
import subprocess
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.manager import InventoryManager
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.file import is_executable
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import PromptVaultSecret, get_file_vault_secret
from ansible.plugins.loader import add_all_plugin_dirs, init_plugin_loader
from ansible.release import __version__
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.path import unfrackpath
from ansible.utils.unsafe_proxy import to_unsafe_text
from ansible.vars.manager import VariableManager
@staticmethod
def setup_vault_secrets(loader, vault_ids, vault_password_files=None, ask_vault_pass=None, create_new_password=False, auto_prompt=True):
    vault_secrets = []
    prompt_formats = {}
    vault_password_files = vault_password_files or []
    if C.DEFAULT_VAULT_PASSWORD_FILE:
        vault_password_files.append(C.DEFAULT_VAULT_PASSWORD_FILE)
    if create_new_password:
        prompt_formats['prompt'] = ['New vault password (%(vault_id)s): ', 'Confirm new vault password (%(vault_id)s): ']
        prompt_formats['prompt_ask_vault_pass'] = ['New Vault password: ', 'Confirm New Vault password: ']
    else:
        prompt_formats['prompt'] = ['Vault password (%(vault_id)s): ']
        prompt_formats['prompt_ask_vault_pass'] = ['Vault password: ']
    vault_ids = CLI.build_vault_ids(vault_ids, vault_password_files, ask_vault_pass, create_new_password, auto_prompt=auto_prompt)
    last_exception = found_vault_secret = None
    for vault_id_slug in vault_ids:
        vault_id_name, vault_id_value = CLI.split_vault_id(vault_id_slug)
        if vault_id_value in ['prompt', 'prompt_ask_vault_pass']:
            built_vault_id = vault_id_name or C.DEFAULT_VAULT_IDENTITY
            prompted_vault_secret = PromptVaultSecret(prompt_formats=prompt_formats[vault_id_value], vault_id=built_vault_id)
            try:
                prompted_vault_secret.load()
            except AnsibleError as exc:
                display.warning('Error in vault password prompt (%s): %s' % (vault_id_name, exc))
                raise
            found_vault_secret = True
            vault_secrets.append((built_vault_id, prompted_vault_secret))
            loader.set_vault_secrets(vault_secrets)
            continue
        display.vvvvv('Reading vault password file: %s' % vault_id_value)
        try:
            file_vault_secret = get_file_vault_secret(filename=vault_id_value, vault_id=vault_id_name, loader=loader)
        except AnsibleError as exc:
            display.warning('Error getting vault password file (%s): %s' % (vault_id_name, to_text(exc)))
            last_exception = exc
            continue
        try:
            file_vault_secret.load()
        except AnsibleError as exc:
            display.warning('Error in vault password file loading (%s): %s' % (vault_id_name, to_text(exc)))
            last_exception = exc
            continue
        found_vault_secret = True
        if vault_id_name:
            vault_secrets.append((vault_id_name, file_vault_secret))
        else:
            vault_secrets.append((C.DEFAULT_VAULT_IDENTITY, file_vault_secret))
        loader.set_vault_secrets(vault_secrets)
    if last_exception and (not found_vault_secret):
        raise last_exception
    return vault_secrets