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
def initialize_locale():
    """Set the locale to the users default setting and ensure
    the locale and filesystem encoding are UTF-8.
    """
    try:
        locale.setlocale(locale.LC_ALL, '')
        dummy, encoding = locale.getlocale()
    except (locale.Error, ValueError) as e:
        raise SystemExit('ERROR: Ansible could not initialize the preferred locale: %s' % e)
    if not encoding or encoding.lower() not in ('utf-8', 'utf8'):
        raise SystemExit('ERROR: Ansible requires the locale encoding to be UTF-8; Detected %s.' % encoding)
    fs_enc = sys.getfilesystemencoding()
    if fs_enc.lower() != 'utf-8':
        raise SystemExit('ERROR: Ansible requires the filesystem encoding to be UTF-8; Detected %s.' % fs_enc)