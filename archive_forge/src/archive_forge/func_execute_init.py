from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import yaml
import shlex
import subprocess
from collections.abc import Mapping
from ansible import context
import ansible.plugins.loader as plugin_loader
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.config.manager import ConfigManager, Setting
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.six import string_types
from ansible.parsing.quoting import is_quoted
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
def execute_init(self):
    """Create initial configuration"""
    seen = {}
    data = []
    config_entries = self._list_entries_from_args()
    plugin_types = config_entries.pop('PLUGINS', None)
    if context.CLIARGS['format'] == 'ini':
        sections = self._get_settings_ini(config_entries, seen)
        if plugin_types:
            for ptype in plugin_types:
                plugin_sections = self._get_settings_ini(plugin_types[ptype], seen)
                for s in plugin_sections:
                    if s in sections:
                        sections[s].extend(plugin_sections[s])
                    else:
                        sections[s] = plugin_sections[s]
        if sections:
            for section in sections.keys():
                data.append('[%s]' % section)
                for key in sections[section]:
                    data.append(key)
                    data.append('')
                data.append('')
    elif context.CLIARGS['format'] in ('env', 'vars'):
        data = self._get_settings_vars(config_entries, context.CLIARGS['format'])
        if plugin_types:
            for ptype in plugin_types:
                for plugin in plugin_types[ptype].keys():
                    data.extend(self._get_settings_vars(plugin_types[ptype][plugin], context.CLIARGS['format']))
    self.pager(to_text('\n'.join(data), errors='surrogate_or_strict'))