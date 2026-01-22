from __future__ import (absolute_import, division, print_function)
from yaml.constructor import SafeConstructor, ConstructorError
from yaml.nodes import MappingNode
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.parsing.yaml.objects import AnsibleMapping, AnsibleSequence, AnsibleUnicode, AnsibleVaultEncryptedUnicode
from ansible.parsing.vault import VaultLib
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var
def construct_yaml_unsafe(self, node):
    try:
        constructor = getattr(node, 'id', 'object')
        if constructor is not None:
            constructor = getattr(self, 'construct_%s' % constructor)
    except AttributeError:
        constructor = self.construct_object
    value = constructor(node)
    return wrap_var(value)