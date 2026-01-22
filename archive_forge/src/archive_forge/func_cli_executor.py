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
@classmethod
def cli_executor(cls, args=None):
    if args is None:
        args = sys.argv
    try:
        display.debug('starting run')
        ansible_dir = Path(C.ANSIBLE_HOME).expanduser()
        try:
            ansible_dir.mkdir(mode=448)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                display.warning("Failed to create the directory '%s': %s" % (ansible_dir, to_text(exc, errors='surrogate_or_replace')))
        else:
            display.debug("Created the '%s' directory" % ansible_dir)
        try:
            args = [to_text(a, errors='surrogate_or_strict') for a in args]
        except UnicodeError:
            display.error('Command line args are not in utf-8, unable to continue.  Ansible currently only understands utf-8')
            display.display(u'The full traceback was:\n\n%s' % to_text(traceback.format_exc()))
            exit_code = 6
        else:
            cli = cls(args)
            exit_code = cli.run()
    except AnsibleOptionsError as e:
        cli.parser.print_help()
        display.error(to_text(e), wrap_text=False)
        exit_code = 5
    except AnsibleParserError as e:
        display.error(to_text(e), wrap_text=False)
        exit_code = 4
    except AnsibleError as e:
        display.error(to_text(e), wrap_text=False)
        exit_code = 1
    except KeyboardInterrupt:
        display.error('User interrupted execution')
        exit_code = 99
    except Exception as e:
        if C.DEFAULT_DEBUG:
            raise
        have_cli_options = bool(context.CLIARGS)
        display.error('Unexpected Exception, this is probably a bug: %s' % to_text(e), wrap_text=False)
        if not have_cli_options or (have_cli_options and context.CLIARGS['verbosity'] > 2):
            log_only = False
            if hasattr(e, 'orig_exc'):
                display.vvv('\nexception type: %s' % to_text(type(e.orig_exc)))
                why = to_text(e.orig_exc)
                if to_text(e) != why:
                    display.vvv('\noriginal msg: %s' % why)
        else:
            display.display('to see the full traceback, use -vvv')
            log_only = True
        display.display(u'the full traceback was:\n\n%s' % to_text(traceback.format_exc()), log_only=log_only)
        exit_code = 250
    sys.exit(exit_code)