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
def process_deprecation(deprecation, top_level=False):
    collection_name = 'removed_from_collection' if top_level else 'collection_name'
    if not isinstance(deprecation, MutableMapping):
        return
    if (is_module or top_level) and 'removed_in' in deprecation:
        callback(deprecation, 'removed_in', collection_name)
    if 'removed_at_date' in deprecation:
        callback(deprecation, 'removed_at_date', collection_name)
    if not (is_module or top_level) and 'version' in deprecation:
        callback(deprecation, 'version', collection_name)