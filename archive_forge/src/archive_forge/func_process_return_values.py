from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping, MutableSet, MutableSequence
from pathlib import Path
from ansible import constants as C
from ansible.release import __version__ as ansible_version
from ansible.errors import AnsibleError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.plugin_docs import read_docstring
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def process_return_values(return_values):
    for return_value in return_values.values():
        if not isinstance(return_value, MutableMapping):
            continue
        if 'version_added' in return_value:
            callback(return_value, 'version_added', 'version_added_collection')
        if isinstance(return_value.get('contains'), MutableMapping):
            process_return_values(return_value['contains'])