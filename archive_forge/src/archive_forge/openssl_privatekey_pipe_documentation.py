from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.plugin_utils.action_module import ActionModuleBase
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey import (
Serialize the object into a dictionary.