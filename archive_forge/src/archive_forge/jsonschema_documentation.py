from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import to_list
from ansible_collections.ansible.utils.plugins.plugin_utils.base.validate import ValidateBase
Std entry point for a validate execution

        :return: Errors or parsed text as structured data
        :rtype: dict

        :example:

        The parse function of a parser should return a dict:
        {"errors": [a list of errors]}
        or
        {"parsed": obj}
        