from __future__ import (absolute_import, division, print_function)
import difflib
import json
import re
import sys
import textwrap
from collections import OrderedDict
from collections.abc import MutableMapping
from copy import deepcopy
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import text_type
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.parsing.yaml.objects import AnsibleUnicode
from ansible.plugins import AnsiblePlugin
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import AnsibleUnsafeText, NativeJinjaUnsafeText, _is_unsafe
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
import yaml
@staticmethod
def host_label(result):
    """Return label for the hostname (& delegated hostname) of a task
        result.
        """
    label = '%s' % result._host.get_name()
    if result._task.delegate_to and result._task.delegate_to != result._host.get_name():
        label += ' -> %s' % result._task.delegate_to
        ahost = result._result.get('_ansible_delegated_vars', {}).get('ansible_host', result._task.delegate_to)
        if result._task.delegate_to != ahost:
            label += '(%s)' % ahost
    return label