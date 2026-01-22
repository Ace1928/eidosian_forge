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
def remove_current_collection_from_versions_and_dates(fragment, collection_name, is_module, return_docs=False):

    def remove(options, option, collection_name_field):
        if options.get(collection_name_field) == collection_name:
            del options[collection_name_field]
    _process_versions_and_dates(fragment, is_module, return_docs, remove)